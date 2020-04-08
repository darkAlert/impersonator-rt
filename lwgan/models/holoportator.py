import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .models import BaseModel
from networks.networks import NetworksFactory, HumanModelRecovery
from utils.nmr import SMPLRenderer
from utils.detectors import PersonMaskRCNNDetector
import utils.cv_utils as cv_utils
import utils.util as util


class Holoportator(BaseModel):
    def __init__(self, opt):
        super(Holoportator, self).__init__(opt)
        self._name = 'Holoportator'

        self._create_networks()

        # prefetch variables
        self.src_info = None
        self.tsf_info = None
        self.first_cam = None


    def _create_networks(self):
        # 1. create generator
        self.generator = self._create_generator().cuda()

        # 2. create hmr
        self.hmr = self._create_hmr().cuda()

        # 3. create render
        self.render = SMPLRenderer(image_size=self._opt.image_size, tex_size=self._opt.tex_size,
                                   has_front=self._opt.front_warp, fill_back=False).cuda()
        # 4. pre-processor
        if self._opt.has_detector:
            self.detector = PersonMaskRCNNDetector(ks=self._opt.bg_ks, threshold=0.5, to_gpu=True)
        else:
            self.detector = None


    def _create_generator(self):
        net = NetworksFactory.get_by_name(self._opt.gen_name, src_dim=3+self._G_cond_nc,
                                          tsf_dim=3+self._G_cond_nc, repeat_num=self._opt.repeat_num)

        if self._opt.load_path:
            self._load_params(net, self._opt.load_path, remove_bg_model=True)
        elif self._opt.load_epoch > 0:
            self._load_network(net, 'G', self._opt.load_epoch)
        else:
            raise ValueError('load_path {} is empty and load_epoch {} is 0'.format(
                self._opt.load_path, self._opt.load_epoch))

        net.eval()
        return net


    def _create_hmr(self):
        hmr = HumanModelRecovery(self._opt.smpl_model)
        saved_data = torch.load(self._opt.hmr_model)
        hmr.load_state_dict(saved_data)
        hmr.eval()
        return hmr


    @torch.no_grad()
    def personalize(self, src_path, src_smpl=None, output_path=''):

        ori_img = cv_utils.read_cv2_img(src_path)

        # resize image and convert the color space from [0, 255] to [-1, 1]
        img = cv_utils.transform_img(ori_img, self._opt.image_size, transpose=True) * 2 - 1.0
        img = torch.tensor(img, dtype=torch.float32).cuda()[None, ...]

        if src_smpl is None:
            img_hmr = cv_utils.transform_img(ori_img, 224, transpose=True) * 2 - 1.0
            img_hmr = torch.tensor(img_hmr, dtype=torch.float32).cuda()[None, ...]
            src_smpl = self.hmr(img_hmr)
        elif isinstance(src_smpl, np.ndarray):
            src_smpl = torch.tensor(src_smpl, dtype=torch.float32).cuda().unsqueeze(0)

        # source process, {'theta', 'cam', 'pose', 'shape', 'verts', 'j2d', 'j3d'}
        src_info = self.hmr.get_details(src_smpl)
        src_f2verts, src_fim, src_wim = self.render.render_fim_wim(src_info['cam'], src_info['verts'])
        # src_f2pts = src_f2verts[:, :, :, 0:2]
        src_info['fim'] = src_fim
        src_info['wim'] = src_wim
        src_info['cond'], _ = self.render.encode_fim(src_info['cam'], src_info['verts'], fim=src_fim, transpose=True)
        src_info['f2verts'] = src_f2verts
        src_info['p2verts'] = src_f2verts[:, :, :, 0:2]
        src_info['p2verts'][:, :, :, 1] *= -1

        if self._opt.only_vis:
            src_info['p2verts'] = self.render.get_vis_f2pts(src_info['p2verts'], src_fim)
        # add image to source info
        src_info['img'] = img
        src_info['image'] = ori_img

        # 2. process the src inputs
        if self.detector is not None:
            bbox, body_mask = self.detector.inference(img[0])
            bg_mask = 1 - body_mask
        else:
            # bg is 1, ft is 0
            bg_mask = util.morph(src_info['cond'][:, -1:, :, :], ks=self._opt.bg_ks, mode='erode')
            body_mask = 1 - bg_mask

        ft_mask = 1 - util.morph(src_info['cond'][:, -1:, :, :], ks=self._opt.ft_ks, mode='erode')
        src_inputs = torch.cat([img * ft_mask, src_info['cond']], dim=1)

        src_info['feats'] = self.generator.encode_src(src_inputs)

        self.src_info = src_info

        if output_path:
            cv_utils.save_cv2_img(src_info['image'], output_path, image_size=self._opt.image_size)


    @torch.no_grad()
    def inference_by_smpls(self, tgt_smpls, cam_strategy='smooth', output_dir=None):
        length = len(tgt_smpls)

        outputs = []
        for t in tqdm(range(length)):
            tgt_smpl = tgt_smpls[t] if tgt_smpls is not None else None

            tsf_inputs = self.transfer_params_by_smpl(tgt_smpl, cam_strategy, t=t)
            preds,_ = self.forward(tsf_inputs, self.tsf_info['T'])

            preds = preds[0].permute(1, 2, 0)
            preds = preds.cpu().numpy()
            outputs.append(preds)

            if output_dir is not None:
                cv_utils.save_cv2_img(preds, output_dir[t], normalize=True)

        return outputs


    def swap_smpl(self, src_cam, src_shape, tgt_smpl, cam_strategy='smooth'):
        tgt_cam = tgt_smpl[:, 0:3].contiguous()
        pose = tgt_smpl[:, 3:75].contiguous()

        # TODO, need more tricky ways
        if cam_strategy == 'smooth':

            cam = src_cam.clone()
            delta_xy = tgt_cam[:, 1:] - self.first_cam[:, 1:]
            cam[:, 1:] += delta_xy

        elif cam_strategy == 'source':
            cam = src_cam
        else:
            cam = tgt_cam

        tsf_smpl = torch.cat([cam, pose, src_shape], dim=1)

        return tsf_smpl


    def transfer_params_by_smpl(self, tgt_smpl, cam_strategy='smooth', t=0):
        # get source info
        src_info = self.src_info

        if isinstance(tgt_smpl, np.ndarray):
            tgt_smpl = torch.tensor(tgt_smpl).float().cuda()[None, ...]

        if t == 0 and cam_strategy == 'smooth':
            self.first_cam = tgt_smpl[:, 0:3].clone()

        # get transfer smpl
        tsf_smpl = self.swap_smpl(src_info['cam'], src_info['shape'], tgt_smpl, cam_strategy=cam_strategy)
        # transfer process, {'theta', 'cam', 'pose', 'shape', 'verts', 'j2d', 'j3d'}
        tsf_info = self.hmr.get_details(tsf_smpl)

        tsf_f2verts, tsf_fim, tsf_wim = self.render.render_fim_wim(tsf_info['cam'], tsf_info['verts'])
        # src_f2pts = src_f2verts[:, :, :, 0:2]
        tsf_info['fim'] = tsf_fim
        tsf_info['wim'] = tsf_wim
        tsf_info['cond'], _ = self.render.encode_fim(tsf_info['cam'], tsf_info['verts'], fim=tsf_fim, transpose=True)
        # tsf_info['sil'] = util.morph((tsf_fim != -1).float(), ks=self._opt.ft_ks, mode='dilate')

        T = self.render.cal_bc_transform(src_info['p2verts'], tsf_fim, tsf_wim)
        tsf_img = F.grid_sample(src_info['img'], T)
        tsf_inputs = torch.cat([tsf_img, tsf_info['cond']], dim=1)

        # add target image to tsf info
        tsf_info['tsf_img'] = tsf_img
        tsf_info['T'] = T

        self.tsf_info = tsf_info

        return tsf_inputs

    def view(self, rt, t, output_dir=None):
        # get source info
        src_info = self.src_info
        src_mesh = self.src_info['verts']
        tsf_mesh = self.rotate_trans(rt, t, src_mesh)

        tsf_f2verts, tsf_fim, tsf_wim = self.render.render_fim_wim(src_info['cam'], tsf_mesh)
        tsf_cond, _ = self.render.encode_fim(src_info['cam'], tsf_mesh, fim=tsf_fim, transpose=True)

        T = self.render.cal_bc_transform(src_info['p2verts'], tsf_fim, tsf_wim)
        tsf_img = F.grid_sample(src_info['img'], T)
        tsf_inputs = torch.cat([tsf_img, tsf_cond], dim=1)

        preds, tsf_mask = self.forward(tsf_inputs, T)

        if self._opt.front_warp:
            preds = self.warp_front(preds, tsf_mask)

        preds = preds[0].permute(1, 2, 0)
        preds = preds.cpu().detach().numpy()

        if output_dir is not None:
            cv_utils.save_cv2_img(preds, output_dir, normalize=True)

        return preds


    def forward(self, tsf_inputs, T):
        src_encoder_outs, src_resnet_outs = self.src_info['feats']

        tsf_color, tsf_mask = self.generator.inference(src_encoder_outs, src_resnet_outs, tsf_inputs, T)
        pred_imgs = (1 - tsf_mask) * tsf_color

        if self._opt.front_warp:
            pred_imgs = self.warp_front(pred_imgs, tsf_mask)

        return pred_imgs, tsf_mask


    def warp_front(self, preds, mask):
        front_mask = self.render.encode_front_fim(self.tsf_info['fim'], transpose=True, front_fn=True)
        preds = (1 - front_mask) * preds + self.tsf_info['tsf_img'] * front_mask * (1 - mask)
        # preds = torch.clamp(preds + self.tsf_info['tsf_img'] * front_mask, -1, 1)
        return preds

    def rotate_trans(self, rt, t, X):
        R = cv_utils.euler2matrix(rt)    # (3 x 3)

        R = torch.FloatTensor(R)[None, :, :].cuda()
        t = torch.FloatTensor(t)[None, None, :].cuda()

        # (bs, Nv, 3) + (bs, 1, 3)
        return torch.bmm(X, R) + t