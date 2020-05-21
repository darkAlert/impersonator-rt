import os.path
import glob
import torchvision.transforms as transforms
import numpy as np
from lwganrt.data.dataset import DatasetBase
from lwganrt.utils import cv_utils
from lwganrt.utils.util import load_pickle_file, ToTensor, ImageTransformer


__all__ = ['AdaBaseDataset', 'AdaDataset']

'''
    The dataset for adaptive training in real-time
'''

class AdaBaseDataset(DatasetBase):
    def __init__(self, opt, is_for_train):
        super(AdaBaseDataset, self).__init__(opt, is_for_train)
        self._name = 'AdaBaseDataset'

        self._intervals = opt.intervals

        # read dataset
        self._read_dataset_paths()

    def __getitem__(self, index):
        # get sample data
        v_info = self._vids_info[index % self._num_videos]
        images, smpls = self._load_pairs(v_info)

        # pack data
        sample = {
            'images': images,
            'smpls': smpls
        }

        sample = self._transform(sample)

        return sample

    def __len__(self):
        return self._dataset_size

    def _read_dataset_paths(self):
        self._root = self._opt.data_dir
        self._vids_dir = os.path.join(self._root, self._opt.images_folder)
        self._smpls_dir = os.path.join(self._root, self._opt.smpls_folder)

        # read video list
        self._num_videos = 0
        self._dataset_size = 0
        self._vids_info = self._read_vids_info()

    def _read_vids_info(self):
        images_path = glob.glob(os.path.join(self._vids_dir, '*'))
        images_path.sort()
        smpls_path = glob.glob(os.path.join(self._smpls_dir, '*'))
        smpls_path.sort()

        assert len(images_path) == len(smpls_path), '{} != {}'.format(len(images_path), len(smpls_path))

        vids_info = [{
            'images': images_path,
            'smpls': smpls_path,
            'length': len(images_path)
        }]

        self._dataset_size += info['length'] // self._intervals
        self._num_videos += 1

        return vids_info

    @property
    def video_info(self):
        return self._vids_info

    def _load_pairs(self, vid_info):
        raise NotImplementedError

    def _create_transform(self):
        transform_list = [
            ImageTransformer(output_size=self._opt.image_size),
            ToTensor()]
        self._transform = transforms.Compose(transform_list)


class AdaDataset(AdaBaseDataset):

    def __init__(self, opt, is_for_train):
        super(AdaDataset, self).__init__(opt, is_for_train)
        self._name = 'AdaDataset'

    def _load_pairs(self, vid_info):
        length = vid_info['length']
        images_path = vid_info['images']
        smpls_path = vid_info['smpls']

        pari_ids = [np.random.randint(0, length), np.random.randint(0, length)]

        # Load data:
        images, smpls = [], []
        for t in pari_ids:
            image_path = images_path[t]
            image = cv_utils.read_cv2_img(image_path)
            images.append(image)

            smpl_path = smpls_path[t]
            smpl = load_pickle_file(smpl_path)
            smpls.append()
        smpls = np.concatenate(smpls, axis=0)

        print ('smpls:', smpls.shape)

        return images, smpls


