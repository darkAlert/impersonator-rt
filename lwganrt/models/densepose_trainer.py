import torch
from .models import BaseModel
from lwganrt.networks.networks import NetworksFactory, HumanModelRecovery, Vgg19, VGGLoss, FaceLoss, StyleLoss
from lwganrt.utils.nmr import SMPLRenderer
import lwganrt.utils.util as util


class BodyRecoveryFlowDP(torch.nn.Module):
    def __init__(self, opt, device=None):
        super(BodyRecoveryFlowDP, self).__init__()
        self._name = 'BodyRecoveryFlowDP'
        self._opt = opt
        self.device = device

        # create networks
        self._init_create_networks()

    def _create_hmr(self):
        hmr = HumanModelRecovery(smpl_pkl_path=self._opt.smpl_model, device=self.device)
        saved_data = torch.load(self._opt.hmr_model)
        hmr.load_state_dict(saved_data)
        hmr.eval()
        return hmr

    def _create_render(self):
        render = SMPLRenderer(map_name=self._opt.map_name,
                               face_path=self._opt.smpl_faces,
                               uv_map_path=self._opt.uv_mapping,
                               image_size=self._opt.image_size,
                               tex_size=self._opt.tex_size,
                               has_front=False,
                               fill_back=False,
                               part_info=self._opt.part_info,
                               front_info=self._opt.front_info,
                               head_info=self._opt.head_info,
                               anti_aliasing=True,
                               background_color=(0, 0, 0),
                               device=self.device)

        return render

    def _init_create_networks(self):
        # hmr and render
        self._hmr = self._create_hmr()
        self._render = self._create_render()

    def forward(self, src_img, src_smpl, use_mask=False):
        # get smpl information
        src_info = self._hmr.get_details(src_smpl)

        # process source inputs
        src_f2verts, src_fim, _ = self._render.render_fim_wim(src_info['cam'], src_info['verts'])
        src_f2verts = src_f2verts[:, :, :, 0:2]
        src_f2verts[:, :, :, 1] *= -1
        src_cond, _ = self._render.encode_fim(src_info['cam'], src_info['verts'], fim=src_fim, transpose=True)
        src_crop_mask = util.morph(src_cond[:, -1:, :, :], ks=3, mode='erode')

        # src input
        if use_mask:
            input_G_src = torch.cat([src_img * (1 - src_crop_mask), src_cond], dim=1)  # DELETE src_crop_mask ???
        else:
            input_G_src = torch.cat([src_img, src_cond], dim=1)

        return input_G_src, src_crop_mask


class DensePose(BaseModel):
    def __init__(self, opt):
        super(DensePose, self).__init__(opt)
        self._name = 'DensePose'
        self.device = torch.device('cuda:' + str(opt.gpu_ids))

        # create networks
        self._init_create_networks()

        # init train variables and losses
        if self._is_train:
            self._init_train_vars()
            self._init_losses()

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()
        elif self._opt.load_path != 'None':
            self._load_params(self._G, self._opt.load_path, need_module=True)

        # prefetch variables
        self._init_prefetch_inputs()

    def _init_create_networks(self):
        # body recovery Flow
        self._bdr = BodyRecoveryFlowDP(opt=self._opt, device=self.device)
        self._bdr.eval()
        self._bdr.cuda()

        # generator network
        self._G = self._create_generator()
        self._G.init_weights()
        self._G.cuda()

    def _create_generator(self):
        return NetworksFactory.get_by_name(self._opt.gen_name, src_dim=3+self._G_cond_nc,
                                           tsf_dim=3+self._G_cond_nc, repeat_num=self._opt.repeat_num,
                                           device=self.device)

    def _init_train_vars(self):
        self._current_lr_G = self._opt.lr_G

        # initialize optimizers
        self._optimizer_G = torch.optim.Adam(self._G.parameters(), lr=self._current_lr_G,
                                             betas=(self._opt.G_adam_b1, self._opt.G_adam_b2))

    def _init_prefetch_inputs(self):
        self._real_src = None
        self._real_uv = None
        self._real_mask = None
        self._input_G_src = None

    def _init_losses(self):
        # define loss functions
        self._crt_smooth_l1 = torch.nn.SmoothL1Loss()
        self._crt_softmax = torch.nn.CrossEntropyLoss()

        # init losses G
        self._loss_g_smooth_l1= self._Tensor([0])
        self._loss_g_softmax= self._Tensor([0])

    def set_input(self, input):

        with torch.no_grad():
            images = input['images']
            smpls = input['smpls']
            uvs = input['uvs']
            src_img = images[:, 0, ...].cuda()
            src_smpl = smpls[:, 0, ...].cuda()
            src_uv = None
            src_mask = None

            input_G_src, src_crop_mask = self._bdr(src_img, src_smpl)

            self._real_src = src_img
            self._real_uv = src_uv
            self._real_mask = src_mask
            self._input_G_src = input_G_src

    def set_train(self):
        self._G.train()
        self._is_train = True

    def set_eval(self):
        self._G.eval()
        self._is_train = False

    def forward(self, keep_data_for_visuals=None):
        # generate fake images
        pred_uvs, pred_masks, debug_data = self._G.forward(self._input_G_src)

        return pred_uvs, pred_masks, debug_data

    def optimize_parameters(self, keep_data_for_visuals=None, trainable=True):
        if self._is_train:
            # run inference
            pred_uvs, pred_masks, debug_data = self.forward()

            loss_G = self._optimize_G(pred_uvs, pred_masks)

            self._optimizer_G.zero_grad()
            loss_G.backward()
            self._optimizer_G.step()

    def _optimize_G(self, pred_uvs, pred_masks):
        self._loss_g_smooth_l1 = self._crt_smooth_l1(pred_uvs, self._real_uv) * self._opt.lambda_rec
        self._loss_g_softmax = self._crt_softmax(pred_masks, self._real_mask) * self._opt.lambda_mask

        # combine losses
        return self._loss_g_smooth_l1 + self._loss_g_softmax

    def save(self, label, save_optimizer=True):
        # save networks
        self._save_network(self._G, 'G', label)

        # save optimizers
        if save_optimizer:
            self._save_optimizer(self._optimizer_G, 'G', label)

    def load(self):
        load_epoch = self._opt.load_epoch

        # load G
        self._load_network(self._G, 'G', load_epoch, need_module=True)

        if self._is_train:
            # load optimizers
            self._load_optimizer(self._optimizer_G, 'G', load_epoch)

    def update_learning_rate(self):
        # updated learning rate G
        final_lr = self._opt.final_lr

        lr_decay_G = (self._opt.lr_G - final_lr) / self._opt.nepochs_decay
        self._current_lr_G -= lr_decay_G
        for param_group in self._optimizer_G.param_groups:
            param_group['lr'] = self._current_lr_G
        print('update G learning rate: %f -> %f' % (self._current_lr_G + lr_decay_G, self._current_lr_G))






