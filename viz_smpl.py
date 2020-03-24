import os
import numpy as np
import cv2
import torch
from networks.networks import HumanModelRecovery
from utils.nmr import SMPLRenderer
from holo.data_struct import DataStruct


class BodyRecoveryFlowH():
    def __init__(self, opt):
        super(BodyRecoveryFlowH, self).__init__()
        self._name = 'BodyRecoveryFlowH'
        self._opt = opt

        # create networks
        self._init_create_networks()

    def _create_hmr(self):
        hmr = HumanModelRecovery(smpl_pkl_path=self._opt.smpl_model)
        saved_data = torch.load(self._opt.hmr_model)
        hmr.load_state_dict(saved_data)
        hmr.eval()
        return hmr

    def _create_render(self):
        render = SMPLRenderer(map_name=self._opt.map_name,
                              uv_map_path=self._opt.uv_mapping,
                              tex_size=self._opt.tex_size,
                              image_size=self._opt.image_size, fill_back=False,
                              anti_aliasing=True, background_color=(0, 0, 0), has_front=False)

        return render

    def _init_create_networks(self):
        # hmr and render
        self._hmr = self._create_hmr().to(self._opt.device)
        self._render = self._create_render().to(self._opt.device)

    def cal_head_bbox(self, kps):
        """
        Args:
            kps: (N, 19, 2)

        Returns:
            bbox: (N, 4)
        """
        NECK_IDS = 12

        image_size = self._opt.image_size

        kps = (kps + 1) / 2.0

        necks = kps[:, NECK_IDS, 0]
        zeros = torch.zeros_like(necks)
        ones = torch.ones_like(necks)

        # min_x = int(max(0.0, np.min(kps[HEAD_IDS:, 0]) - 0.1) * image_size)
        min_x, _ = torch.min(kps[:, NECK_IDS:, 0] - 0.05, dim=1)
        min_x = torch.max(min_x, zeros)

        max_x, _ = torch.max(kps[:, NECK_IDS:, 0] + 0.05, dim=1)
        max_x = torch.min(max_x, ones)

        # min_x = int(max(0.0, np.min(kps[HEAD_IDS:, 0]) - 0.1) * image_size)
        min_y, _ = torch.min(kps[:, NECK_IDS:, 1] - 0.05, dim=1)
        min_y = torch.max(min_y, zeros)

        max_y, _ = torch.max(kps[:, NECK_IDS:, 1], dim=1)
        max_y = torch.min(max_y, ones)

        min_x = (min_x * image_size).long()  # (T, 1)
        max_x = (max_x * image_size).long()  # (T, 1)
        min_y = (min_y * image_size).long()  # (T, 1)
        max_y = (max_y * image_size).long()  # (T, 1)

        # print(min_x.shape, max_x.shape, min_y.shape, max_y.shape)
        rects = torch.stack((min_x, max_x, min_y, max_y), dim=1)
        # import ipdb
        # ipdb.set_trace()
        return rects


class BDROptions():
    def __init__(self, device):
        self.smpl_model = 'assets/pretrains/smpl_model.pkl'
        self.hmr_model = 'assets/pretrains/hmr_tf2pt.pth'
        self.map_name = 'uv_seg'
        self.uv_mapping = 'assets/pretrains/mapper.txt'
        self.tex_size = 3
        self.image_size = 256
        self.device = device


def get_file_paths(path):
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]:
            file_paths.append(os.path.join(dirpath,filename))
            file_paths.sort()

    return file_paths

def project_smpl_onto_image(root_dir, img_dir, smpl_path, output_dir, frame_id=0):
    img_dir = os.path.join(root_dir, img_dir)
    smpl_path = os.path.join(root_dir, smpl_path)
    output_dir = os.path.join(root_dir, output_dir)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    # Parse image paths:
    img_paths = get_file_paths(img_dir)

    # Load smpl:
    with np.load(smpl_path, encoding='latin1', allow_pickle=True) as data:
        smpl_data = dict(data)
    cams = smpl_data['cams'][frame_id]
    pose = smpl_data['pose'][frame_id]
    shape = smpl_data['shape'][frame_id]
    smpl_vec = np.concatenate((cams, pose, shape), axis=0)
    smpl_vec = np.expand_dims(smpl_vec, axis=0)
    smpl = torch.from_numpy(smpl_vec.astype(np.float32)).to(device)
    print (smpl.shape)


    # Init BDR:
    opt = BDROptions(device)
    bdr = BodyRecoveryFlowH(opt)
    src_info = bdr._hmr.get_details(smpl)
    src_f2verts, src_fim, _ = bdr._render.render_fim_wim(src_info['cam'], src_info['verts'])

    bbox = bdr.cal_head_bbox(src_info['j2d'])

    print ('bbox:',bbox)

    img = src_fim.permute(1,2,0).cpu().numpy()
    path = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/smpl.png'
    cv2.imwrite(path, img)

    # print (src_info)

def check_head_bboxes(smpl_dir):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Init BDR:
    opt = BDROptions(device)
    bdr = BodyRecoveryFlowH(opt)

    # Parse data:
    data_smpl = DataStruct().parse(smpl_dir, levels='subject/light/garment/scene/cam', ext='npz')

    for node, path in data_smpl.nodes('cam'):
        print ('Processing', path)

        # Load smpl:
        smpl_path = [smpl.abs_path for smpl in data_smpl.items(node)][0]
        with np.load(smpl_path, encoding='latin1', allow_pickle=True) as data:
            smpl_data = dict(data)
        n = len(smpl_data['cams'])
        for frame_id in range(n):
            cams = smpl_data['cams'][frame_id]
            pose = smpl_data['pose'][frame_id]
            shape = smpl_data['shape'][frame_id]
            smpl_vec = np.concatenate((cams, pose, shape), axis=0)
            smpl_vec = np.expand_dims(smpl_vec, axis=0)
            smpl = torch.from_numpy(smpl_vec.astype(np.float32)).to(device)

            # Compute bounding box:
            src_info = bdr._hmr.get_details(smpl)
            src_f2verts, src_fim, _ = bdr._render.render_fim_wim(src_info['cam'], src_info['verts'])
            bbox = bdr.cal_head_bbox(src_info['j2d'])

            if bbox[0,0] > 255 or bbox[0,1] > 255 or bbox[0,2] > 255 or \
                    bbox[0,3] > 255 or bbox[0,1] - bbox[0,0] <= 0 or bbox[0,3] - bbox[0,2] <= 0:
                print('-----')
                print (smpl_path, frame_id)
                print (bbox)
                print ('-----')


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    root_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data'
    img_dir = 'avatars/person_1/light-100_temp-5600/garment_1/freestyle/cam1'
    smpl_path = 'smpl_aligned_lwgan/person_2/light-100_temp-5600/garments_2/freestyle/cam3/smpl.npz'
    output_dir = 'lwgan_smpl_test'
    project_smpl_onto_image(root_dir, img_dir, smpl_path, output_dir, frame_id=259)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # check_head_bboxes('/home/darkalert/KazendiJob/Data/HoloVideo/Data/smpl_aligned_lwgan')


if __name__ == '__main__':
    main()