import os.path
import torchvision.transforms as transforms
from data.dataset import DatasetBase
import numpy as np
from utils import cv_utils
from utils.util import ToTensor, ImageTransformer


__all__ = ['HoloBaseDataset', 'HoloDataset']


class HoloBaseDataset(DatasetBase):

	def __init__(self, opt, is_for_train):
		super(HoloBaseDataset, self).__init__(opt, is_for_train)
		self._name = 'HoloBaseDataset'

		# read dataset
		self._read_dataset_paths()

	def __getitem__(self, index):
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
				cams_dir = os.path.join(self._vids_dir, line)
				cams_path = [p.path for p in os.scandir(cams_dir) if p.is_dir()]
				cams_path.sort()
				cams_image_path = []
				for cam_path in cams_path:
					images_path = []
					for file in os.listdir(cam_path):
						if any(file.endswith(extension) for extension in self._IMG_EXTENSIONS):
							images_path.append(os.path.join(cam_path,file))
					images_path.sort()
					cams_image_path.append(images_path)
				assert (len(p) == len(cams_image_path[0]) for p in cams_image_path)

				# Load smpls:
				smpl_dir = os.path.join(self._smpls_dir, line)
				cams_path = [p.path for p in os.scandir(smpl_dir) if p.is_dir()]
				cams_path.sort()
				cams_smpl = []
				for c_path in cams_path:
					smpl_path = os.path.join(c_path, 'smpl.npz')
					with np.load(smpl_path, encoding='latin1', allow_pickle=True) as data:
						cams_smpl.append(dict(data))
				assert len(cams_image_path[0]) == cams_smpl[0]['cams'].shape[0], \
					'{} != {}'.format(len(cams_image_path[0]), cams_smpl[0]['cams'].shape[0])

				info = {
					'images': cams_image_path,
					'num_frames': len(cams_image_path[0]),
					'num_cams': len(cams_image_path),
					'smpl': cams_smpl,
				}

				vids_info.append(info)
				self._dataset_size += info['num_frames']
				self._num_videos += 1
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


class HoloDataset(HoloBaseDataset):
	
	def __init__(self, opt, is_for_train):
		super(HoloDataset, self).__init__(opt, is_for_train)
		self._name = 'HoloDataset'

	def _load_pairs(self, vid_info):
		num_frames = vid_info['num_frames']
		num_cams = vid_info['num_cams']

		frame_id = np.random.randint(0, num_frames)
		first_cam_id = np.random.randint(0, num_cams)
		second_cam_id = np.random.randint(0, num_cams)
		while first_cam_id == second_cam_id:
			second_cam_id = np.random.randint(0, num_cams)
		pair_ids = np.array([first_cam_id, second_cam_id], dtype=np.int32)

		# SMPL:
		s1 = vid_info['smpl'][pair_ids[0]]['cams'][frame_id]
		s2 = vid_info['smpl'][pair_ids[0]]['pose'][frame_id]
		s3 = vid_info['smpl'][pair_ids[0]]['shape'][frame_id]
		smpls_1 = np.concatenate((s1,s2,s3),axis=0)
		smpls_1 = np.expand_dims(smpls_1, axis=0)

		s1 = vid_info['smpl'][pair_ids[1]]['cams'][frame_id]
		s2 = vid_info['smpl'][pair_ids[1]]['pose'][frame_id]
		s3 = vid_info['smpl'][pair_ids[1]]['shape'][frame_id]
		smpls_2 = np.concatenate((s1, s2, s3), axis=0)
		smpls_2 = np.expand_dims(smpls_2, axis=0)

		smpls = np.concatenate((smpls_1, smpls_2), axis=0)

		images = []
		cams_image_path = vid_info['images']
		for t in pair_ids:
			image_path = cams_image_path[t][frame_id]
			image = cv_utils.read_cv2_img(image_path)
			images.append(image)

		return images, smpls