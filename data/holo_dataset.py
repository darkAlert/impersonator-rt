import os.path
import torchvision.transforms as transforms
from data.dataset import DatasetBase
import numpy as np
from utils import cv_utils
from utils.util import ToTensor, ImageTransformer
from enum import Enum

__all__ = ['HoloBaseDataset', 'HoloDataset']


class HoloBaseDataset(DatasetBase):
	class Mode(Enum):
		NOVEL_VIEW = 1
		MOTION_IMIT = 2

	def __init__(self, opt, is_for_train):
		super(HoloBaseDataset, self).__init__(opt, is_for_train)
		self._name = 'HoloBaseDataset'
		self._intervals = opt.holo_intervals

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
		self._root = self._opt.holo_data_dir
		self._vids_dir = os.path.join(self._root, self._opt.holo_images_folder)
		self._smpls_dir = os.path.join(self._root, self._opt.holo_smpls_folder)

		# read video list
		self._num_videos = 0
		self._dataset_size = 0
		use_ids_filename = self._opt.holo_train_ids_file if self._is_for_train else self._opt.holo_test_ids_file
		use_ids_filepath = os.path.join(self._root, use_ids_filename)
		self._vids_info = self._read_vids_info(use_ids_filepath, mode=self.Mode.NOVEL_VIEW)
		self._vids_info = self._vids_info + self._read_vids_info(use_ids_filepath, mode=self.Mode.MOTION_IMIT)

	def _read_vids_info(self, file_path, mode=Mode.NOVEL_VIEW):
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
					'mode': mode
				}

				vids_info.append(info)
				self._num_videos += 1
				if mode == self.Mode.NOVEL_VIEW:
					self._dataset_size += info['num_frames']
				elif mode == self.Mode.MOTION_IMIT:
					self._dataset_size += info['num_frames'] // self._intervals
				else:
					raise NotImplementedError
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

		# Select frame and camera ids:
		cam_ids, frame_ids = [], []
		if vid_info['mode'] == self.Mode.NOVEL_VIEW:
			frame_ids.append(np.random.randint(0, num_frames))
			frame_ids.append(frame_ids[-1])
			cam_ids.append(np.random.randint(0, num_cams))
			cam_ids.append(np.random.randint(0, num_cams))
			while cam_ids[0] == cam_ids[1]:
				cam_ids[1] = np.random.randint(0, num_cams)
		elif vid_info['mode'] == self.Mode.MOTION_IMIT:
			frame_ids.append(np.random.randint(0, num_frames))
			frame_ids.append(np.random.randint(0, num_frames))
			cam_ids.append(np.random.randint(0, num_cams))
			cam_ids.append(np.random.randint(0, num_cams))
		else:
			raise NotImplementedError

		# SMPL:
		s1 = vid_info['smpl'][cam_ids[0]]['cams'][frame_ids[0]]
		s2 = vid_info['smpl'][cam_ids[0]]['pose'][frame_ids[0]]
		s3 = vid_info['smpl'][cam_ids[0]]['shape'][frame_ids[0]]
		smpls_1 = np.concatenate((s1,s2,s3),axis=0)
		smpls_1 = np.expand_dims(smpls_1, axis=0)

		s1 = vid_info['smpl'][cam_ids[1]]['cams'][frame_ids[1]]
		s2 = vid_info['smpl'][cam_ids[1]]['pose'][frame_ids[1]]
		s3 = vid_info['smpl'][cam_ids[1]]['shape'][frame_ids[1]]
		smpls_2 = np.concatenate((s1, s2, s3), axis=0)
		smpls_2 = np.expand_dims(smpls_2, axis=0)

		smpls = np.concatenate((smpls_1, smpls_2), axis=0)

		# Images:
		images = []
		cams_image_path = vid_info['images']
		for ci,fi in zip(cam_ids,frame_ids):
			image_path = cams_image_path[ci][fi]
			image = cv_utils.read_cv2_img(image_path)
			images.append(image)

		return images, smpls