import os.path
import torchvision.transforms as transforms
from data.dataset import DatasetBase
import numpy as np
from utils import cv_utils
from utils.util import load_pickle_file, ToTensor, ImageTransformer
import glob


__all__ = ['MyImPerBaseDataset', 'MyImPerDataset']


class MyImPerBaseDataset(DatasetBase):

    def __init__(self, opt, is_for_train):
        super(MyImPerBaseDataset, self).__init__(opt, is_for_train)
        self._name = 'MyImPerBaseDataset'

        self._intervals = opt.intervals

        # read dataset
        self._read_dataset_paths()

    def __getitem__(self, index):
        # assert (index < self._dataset_size)

        # start_time = time.time()
        # get sample data
        v_info = self._vids_info[index % self._num_videos]
        images, smpls = self._load_pairs(v_info)

        # pack data
        sample = {
            'images': images,
            'smpls': smpls
        }

        sample = self._transform(sample)
        # print(time.time() - start_time)

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
        use_ids_filename = self._opt.train_ids_file if self._is_for_train else self._opt.test_ids_file
        use_ids_filepath = os.path.join(self._root, use_ids_filename)
        self._vids_info = self._read_vids_info(use_ids_filepath)

    def _read_vids_info(self, file_path):
        vids_info = []
        with open(file_path, 'r') as reader:

            lines = []
            for line in reader:
                line = line.rstrip()
                lines.append(line)
            total = len(lines)

            for i, line in enumerate(lines):
                if len(line) <= 1:
                    continue
                # Get image paths:
                img_dir = os.path.join(self._vids_dir, line)
                images_path = []
                for file in os.listdir(img_dir):
                    if any(file.endswith(extension) for extension in self._IMG_EXTENSIONS):
                        images_path.append(os.path.join(img_dir, file))
                images_path.sort()

                # Load smpls:
                smpl_dir = os.path.join(self._smpls_dir, line)
                smpl_path = os.path.join(smpl_dir, 'smpl.npz')
                with np.load(smpl_path, encoding='latin1', allow_pickle=True) as data:
                    smpls = dict(data)

                info = {
                    'images': images_path,
                    'smpl': smpls,
                    'length': len(images_path)
                }

                vids_info.append(info)
                self._num_videos += 1
                self._dataset_size += info['length'] // self._intervals
                print('loading video = {}, {} / {}'.format(line, i, total))

                if self._opt.debug:
                    if i > 1:
                        break

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


class MyImPerDataset(MyImPerBaseDataset):

    def __init__(self, opt, is_for_train):
        super(MyImPerDataset, self).__init__(opt, is_for_train)
        self._name = 'MyImPerDataset'

    def _load_pairs(self, vid_info):
        length = vid_info['length']

        # Select frame ids:
        frame_ids = []
        frame_ids.append(np.random.randint(0, length))
        frame_ids.append(np.random.randint(0, length))

        # SMPL:
        s1 = vid_info['smpl']['cams'][frame_ids[0]]
        s2 = vid_info['smpl']['pose'][frame_ids[0]]
        s3 = vid_info['smpl']['shape'][frame_ids[0]]
        smpls_1 = np.concatenate((s1, s2, s3), axis=0)
        smpls_1 = np.expand_dims(smpls_1, axis=0)

        s1 = vid_info['smpl']['cams'][frame_ids[1]]
        s2 = vid_info['smpl']['pose'][frame_ids[1]]
        s3 = vid_info['smpl']['shape'][frame_ids[1]]
        smpls_2 = np.concatenate((s1, s2, s3), axis=0)
        smpls_2 = np.expand_dims(smpls_2, axis=0)

        smpls = np.concatenate((smpls_1, smpls_2), axis=0)

        # Images:
        images = []
        cams_image_path = vid_info['images']
        for fi in frame_ids:
            image_path = cams_image_path[fi]
            image = cv_utils.read_cv2_img(image_path)
            images.append(image)

        return images, smpls


