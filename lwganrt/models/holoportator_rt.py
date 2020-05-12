import torch
import torch.nn.functional as F
import numpy as np
from .models import BaseModel
from lwganrt.networks.networks import NetworksFactory, HumanModelRecovery
from lwganrt.utils.detectors import PersonMaskRCNNDetector
from lwganrt.utils.nmr import SMPLRenderer
import lwganrt.utils.cv_utils as cv_utils
import lwganrt.utils.util as util


class HoloportatorRT(BaseModel):
    def __init__(self, opt, device):
        super(HoloportatorRT, self).__init__(opt)
        self._name = 'HoloportatorRT'
        self.device = device

        self._create_networks()

        # prefetch variables
        self.src_info = None
        self.tsf_info = None


    def _create_networks(self):
        # 1. create generator
        self.generator = self._create_generator()

        # 2. create hmr
        self.hmr = self._create_hmr()

        # 3. create render
        self.render = SMPLRenderer(face_path=self._opt.smpl_faces,
                                   uv_map_path=self._opt.uv_mapping,
                                   image_size=self._opt.image_size,
                                   tex_size=self._opt.tex_size,
                                   has_front=self._opt.front_warp,
                                   fill_back=False,
                                   part_info=self._opt.part_info,
                                   front_info=self._opt.front_info,
                                   head_info=self._opt.head_info,
                                   device = self.device)

        # 4. pre-processor
        if self._opt.has_detector:
            self.detector = PersonMaskRCNNDetector(ks=self._opt.bg_ks, threshold=0.5, device=self.device)
        else:
            self.detector = None

    def _create_generator(self):
        net = NetworksFactory.get_by_name(self._opt.gen_name, src_dim=3+self._G_cond_nc,
                                          tsf_dim=3+self._G_cond_nc, repeat_num=self._opt.repeat_num,
                                          device=self.device)
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
        hmr = HumanModelRecovery(self._opt.smpl_model, device=self.device)
        saved_data = torch.load(self._opt.hmr_model)
        hmr.load_state_dict(saved_data)
        hmr.eval()

        return hmr


    @torch.no_grad()
    def personalize(self, src_img, src_smpl):
        # source process, {'theta', 'cam', 'pose', 'shape', 'verts', 'j2d', 'j3d'}
        src_info = self.hmr.get_details(src_smpl)
        src_f2verts, src_fim, src_wim = self.render.render_fim_wim(src_info['cam'], src_info['verts'])
        src_info['fim'] = src_fim
        src_info['wim'] = src_wim
        src_info['cond'], _ = self.render.encode_fim(src_info['cam'], src_info['verts'], fim=src_fim, transpose=True)
        src_info['f2verts'] = src_f2verts
        src_info['p2verts'] = src_f2verts[:, :, :, 0:2]
        src_info['p2verts'][:, :, :, 1] *= -1

        # add image to source info
        src_info['img'] = src_img

        # get front mask:
        if self.detector is not None:
            bbox, ft_mask = self.detector.inference(src_img[0])
        else:
            ft_mask = 1 - util.morph(src_info['cond'][:, -1:, :, :], ks=self._opt.ft_ks, mode='erode')

        src_inputs = torch.cat([src_img * ft_mask, src_info['cond']], dim=1)

        src_info['feats'] = self.generator.encode_src(src_inputs)

        self.src_info = src_info

    @torch.no_grad()
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

        preds = preds[0].permute(1, 2, 0)
        preds = preds.cpu().detach().numpy()

        if output_dir is not None:
            cv_utils.save_cv2_img(preds, output_dir, normalize=True)

        return preds

    @torch.no_grad()
    def inference(self, tgt_smpl, cam_strategy='smooth', output_dir=None):
        # get target info
        self.src_info['cam'] = tgt_smpl[:, 0:3].contiguous()

        tsf_inputs = self.transfer_params_by_smpl(tgt_smpl, cam_strategy)
        preds,_ = self.forward(tsf_inputs, self.tsf_info['T'])

        preds = preds.permute(0, 2, 3, 1)
        preds = preds.cpu().detach().numpy()

        if output_dir is not None:
            cv_utils.save_cv2_img(preds, output_dir, normalize=True)

        return preds

    def forward(self, tsf_inputs, T):
        src_encoder_outs, src_resnet_outs = self.src_info['feats']

        tsf_color, tsf_mask = self.generator.inference(src_encoder_outs, src_resnet_outs, tsf_inputs, T)
        pred_imgs = (1 - tsf_mask) * tsf_color

        return pred_imgs, tsf_mask

    def rotate_trans(self, rt, t, X):
        R = cv_utils.euler2matrix(rt)    # (3 x 3)

        R = torch.FloatTensor(R)[None, :, :].to(self.device)
        t = torch.FloatTensor(t)[None, None, :].to(self.device)

        # (bs, Nv, 3) + (bs, 1, 3)
        return torch.bmm(X, R) + t

    def transfer_params_by_smpl(self, tgt_smpl, cam_strategy='smooth', t=0):
        # get source info
        src_info = self.src_info

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


def prepare_input(img, smpl, image_size=256, device=None):
    # resize image and convert the color space from [0, 255] to [-1, 1]
    if isinstance(img, np.ndarray):
        prep_img = cv_utils.transform_img(img, image_size, transpose=True) * 2 - 1.0
        prep_img = torch.tensor(prep_img, dtype=torch.float32).unsqueeze(0)
    else:
        raise NotImplementedError

    if isinstance(smpl, np.ndarray):
        if smpl.ndim == 1:
            prep_smpl = torch.tensor(smpl, dtype=torch.float32).unsqueeze(0)
        else:
            prep_smpl = torch.tensor(smpl, dtype=torch.float32)
    else:
        raise NotImplementedError

    if device is not None:
        prep_img = prep_img.to(device)
        prep_smpl = prep_smpl.to(device)

    return prep_img, prep_smpl