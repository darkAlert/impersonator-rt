import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from lwganrt.data.dataset import PairSampleDataset
from lwganrt.utils.util import load_pickle_file, write_pickle_file, mkdirs, mkdir, clear_dir
import lwganrt.utils.cv_utils as cv_utils
from lwganrt.utils.detectors import PersonMaskRCNNDetector


@torch.no_grad()
def write_pair_info(src_info, tsf_info, out_file, model):
    """
    Args:
        src_info:
        tsf_info:
        out_file:
        model:
    Returns:

    """
    pair_data = dict()

    pair_data['from_face_index_map'] = src_info['fim'][0][:, :, None].cpu().numpy()
    pair_data['to_face_index_map'] = tsf_info['fim'][0][:, :, None].cpu().numpy()
    pair_data['T'] = tsf_info['T'][0].cpu().numpy()
    pair_data['warp'] = tsf_info['tsf_img'][0].cpu().numpy()
    pair_data['smpls'] = torch.cat([src_info['theta'], tsf_info['theta']], dim=0).cpu().numpy()
    pair_data['j2d'] = torch.cat([src_info['j2d'], tsf_info['j2d']], dim=0).cpu().numpy()

    tsf_f2verts, tsf_fim, tsf_wim = model.render.render_fim_wim(tsf_info['cam'], tsf_info['verts'])
    tsf_p2verts = tsf_f2verts[:, :, :, 0:2]
    tsf_p2verts[:, :, :, 1] *= -1

    T_cycle = model.render.cal_bc_transform(tsf_p2verts, src_info['fim'], src_info['wim'])
    pair_data['T_cycle'] = T_cycle[0].cpu().numpy()

    write_pickle_file(out_file, pair_data)


def scan_tgt_paths(tgt_path, itv=20):
    if os.path.isdir(tgt_path):
        all_tgt_paths = glob.glob(os.path.join(tgt_path, '*'))
        all_tgt_paths.sort()
        all_tgt_paths = all_tgt_paths[::itv]
    else:
        all_tgt_paths = [tgt_path]

    return all_tgt_paths

def load_smpls(smpl_path, itv=20):
    smpl_data = load_pickle_file(smpl_path)
    pose = smpl_data['pose']
    shape = smpl_data['shape']
    cams = smpl_data['cams']
    smpls = []
    for i in range(0,len(pose),itv):
        vec = np.concatenate((cams[i], pose[i], shape[i]), axis=0)
        smpls.append(torch.tensor(vec, dtype=torch.float32).unsqueeze(0))

    return smpls

def meta_imitate(opt, model, save_imgs=True):
    src_path = opt.src_path
    output_dir = opt.output_dir

    all_tgt_smpls = load_smpls(opt.pri_smpl_path, itv=40)

    out_img_dir, out_pair_dir = mkdirs([os.path.join(output_dir, 'imgs'), os.path.join(output_dir, 'pairs')])
    img_pair_list = []

    for t in tqdm(range(len(all_tgt_smpls))):
        tgt_path = opt.pri_path
        tgt_smpl = all_tgt_smpls[t].to(model.device)
        preds = model.inference(tgt_smpl, cam_strategy=opt.cam_strategy)

        if save_imgs:
            tgt_name = os.path.split(tgt_path)[-1]
            out_path = os.path.join(out_img_dir, 'pred_' + tgt_name)
            cv_utils.save_cv2_img(preds[0], out_path, normalize=True)
            write_pair_info(model.src_info, model.tsf_info,
                            os.path.join(out_pair_dir, '{:0>8}.pkl'.format(t)), model=model)

            img_pair_list.append((src_path, tgt_path))

    if save_imgs:
        write_pickle_file(os.path.join(output_dir, 'pairs_meta.pkl'), img_pair_list)


class MetaCycleDataSet(PairSampleDataset):
    def __init__(self, opt):
        super(MetaCycleDataSet, self).__init__(opt, True)
        self._name = 'MetaCycleDataSet'

    def _read_dataset_paths(self):
        # read pair list
        self._dataset_size = 0
        self._read_samples_info(None, self._opt.pkl_dir, self._opt.pair_ids_filepath)

    def _read_samples_info(self, im_dir, pkl_dir, pair_ids_filepath):
        """
        Args:
            im_dir:
            pkl_dir:
            pair_ids_filepath:

        Returns:

        """
        # 1. load image pair list
        self.im_pair_list = load_pickle_file(pair_ids_filepath)

        # 2. load pkl file paths
        self.all_pkl_paths = sorted(glob.glob((os.path.join(pkl_dir, '*.pkl'))))

        assert len(self.im_pair_list) == len(self.all_pkl_paths), '{} != {}'.format(
            len(self.im_pair_list), len(self.all_pkl_paths)
        )
        self._dataset_size = len(self.im_pair_list)

    def __getitem__(self, item):
        """
        Args:
            item (int):  index of self._dataset_size

        Returns:
            new_sample (dict): items contain
                --src_inputs (torch.FloatTensor): (3+3, h, w)
                --tsf_inputs (torch.FloatTensor): (3+3, h, w)
                --T (torch.FloatTensor): (h, w, 2)
                --head_bbox (torch.IntTensor): (4), hear 4 = [lt_x, lt_y, rt_x, rt_y]
                --valid_bbox (torch.FloatTensor): (1), 1.0 valid and 0.0 invalid.
                --images (torch.FloatTensor): (2, 3, h, w)
                --pseudo_masks (torch.FloatTensor) : (2, 1, h, w)
                --bg_inputs (torch.FloatTensor): (3+1, h, w) or (2, 3+1, h, w) if self.is_both is True
        """
        im_pairs = self.im_pair_list[item]
        pkl_path = self.all_pkl_paths[item]

        sample = self.load_sample(im_pairs, pkl_path)
        sample = self.preprocess(sample)

        sample['preds'] = torch.tensor(self.load_init_preds(im_pairs[1])).float()

        return sample

    def load_init_preds(self, pred_path):
        pred_img_name = os.path.split(pred_path)[-1]
        pred_img_path = os.path.join(self._opt.preds_img_folder, 'pred_' + pred_img_name)

        img = cv_utils.read_cv2_img(pred_img_path)
        img = cv_utils.transform_img(img, self._opt.image_size, transpose=True)
        img = img * 2 - 1

        return img


def make_dataset(opt):
    class Config(object):
        pass

    config = Config()

    output_dir = opt.output_dir

    config.pair_ids_filepath = os.path.join(output_dir, 'pairs_meta.pkl')
    config.pkl_dir = os.path.join(output_dir, 'pairs')
    config.preds_img_folder = os.path.join(output_dir, 'imgs')
    config.image_size = opt.image_size
    config.map_name = opt.map_name
    config.uv_mapping = opt.uv_mapping
    config.is_both = False
    config.bg_ks = opt.bg_ks
    config.ft_ks = opt.ft_ks

    meta_cycle_ds = MetaCycleDataSet(opt=config)
    length = len(meta_cycle_ds)

    data_loader = torch.utils.data.DataLoader(
        meta_cycle_ds,
        batch_size=min(length, opt.batch_size),
        shuffle=False,
        num_workers=4,
        drop_last=True)

    return data_loader

@torch.no_grad()
def detect_and_apply_mask(model_path, src_img, device):
    # Detect mask:
    detector = PersonMaskRCNNDetector(ks=1, threshold=0.5, device=device, pretrained_path=model_path)
    _, ft_mask = detector.inference(src_img[0])

    # Apply mask:
    src_img = (src_img + 1) / 2.0 * ft_mask
    src_img = src_img * 2 - 1.0

    return src_img


def adaptive_personalize(opt, model, src_img, src_smpl):
    clear_dir(opt.output_dir)

    # Detect mask:
    src_img = detect_and_apply_mask(opt.maskrcnn_path, src_img, model.device)

    # Save src image:
    img = src_img.permute(0, 2, 3, 1)[0].cpu().detach().numpy()
    cv_utils.save_cv2_img(img, opt.src_path, normalize=True)

    print('\n\t\t\tPersonalization: meta imitation...')
    model.personalize(src_img, src_smpl)
    meta_imitate(opt, model, save_imgs=True)

    # post tune
    print('\n\t\t\tPersonalization: meta cycle finetune...')
    loader = make_dataset(opt)
    model.post_personalize(loader, verbose=True)