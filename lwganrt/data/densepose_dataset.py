import os.path
import torchvision.transforms as transforms
from lwganrt.data.dataset import DatasetBase
from lwganrt.utils import cv_utils
from lwganrt.utils.util import ToTensorDensePose, ImageTransformer
import numpy as np
import cv2
import pickle


__all__ = ['DensePoseBaseDataset', 'DensePoseDataset']


def load_pickle_file(pkl_path):
	with open(pkl_path, 'rb') as f:
		data = pickle.load(f, encoding='latin1')

	return data

def decode_uv(encoded_uv):
	uv = None
	if encoded_uv is not None:
		uv = cv2.imdecode(encoded_uv, 1)
	return uv


class DensePoseBaseDataset(DatasetBase):
	def __init__(self, opt, is_for_train):
		super(DensePoseBaseDataset, self).__init__(opt, is_for_train)
		self._name = 'DensePoseBaseDataset'

		# read dataset
		self._read_dataset_paths()

	def __getitem__(self, index):
		v_info = self._vids_info[index % self._num_videos]
		image, smpl, uvs, mask = self._load_input(v_info)

		# pack data
		sample = {
			'images': [image],
			'smpl': smpl,
			'uvs': uvs,
			'mask': mask
		}

		sample = self._transform(sample)

		_,h,w,c = sample['images'].shape
		sample['images'] = sample['images'].view(h,w,c)
		sample['mask'] = sample['mask'].long()

		return sample

	def __len__(self):
		return self._dataset_size

	def _read_dataset_paths(self):
		self._root = self._opt.holo_data_dir
		self._vids_dir = os.path.join(self._root, self._opt.holo_images_folder)
		self._smpls_dir = os.path.join(self._root, self._opt.holo_smpls_folder)
		self._uvs_dir = os.path.join(self._root, self._opt.holo_uvs_folder)

		# read video list
		self._num_videos = 0
		self._dataset_size = 0
		use_ids_filename = self._opt.holo_train_ids_file if self._is_for_train else self._opt.holo_test_ids_file
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
				cams_names = []
				for cam_path in cams_path:
					images_path = []
					names = {}
					for file in os.listdir(cam_path):
						if any(file.endswith(extension) for extension in self._IMG_EXTENSIONS):
							images_path.append(os.path.join(cam_path,file))
							name = file.split('.')[0]
							names[name] = True
					images_path.sort()
					cams_image_path.append(images_path)
					cams_names.append(names)
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

				# Load UVs:
				uvs_dir = os.path.join(self._uvs_dir, line)
				cams_path = [p.path for p in os.scandir(uvs_dir) if p.is_dir()]
				cams_path.sort()
				cams_uvs = []
				for cam_i, cam_path in enumerate(cams_path):
					uv_path = []
					for file in os.listdir(cam_path):
						if file.endswith('.pkl') :
							name = file.split('.')[0]
							if name in cams_names[cam_i]:
								uv_path.append(os.path.join(cam_path,file))
					uv_path.sort()
					cams_uvs.append(uv_path)
				assert (len(p) == len(cams_uvs[0]) for p in cams_uvs)
				assert len(cams_image_path[0]) == len(cams_uvs[0]), \
					'{} != {}'.format(len(cams_image_path[0]), len(cams_uvs[0]))

				info = {
					'images': cams_image_path,
					'num_frames': len(cams_image_path[0]),
					'num_cams': len(cams_image_path),
					'smpl': cams_smpl,
					'uvs': cams_uvs,
				}

				vids_info.append(info)
				self._num_videos += 1
				self._dataset_size += info['num_frames']
				print('loading video = {}, {} / {}'.format(line, i, total))

				if self._opt.debug:
					if i > 1:
						break

		return vids_info

	@property
	def video_info(self):
		return self._vids_info

	def _load_input(self, vid_info):
		raise NotImplementedError

	def _create_transform(self):
		transform_list = [
			ImageTransformer(output_size=self._opt.image_size),
			ToTensorDensePose()]
		self._transform = transforms.Compose(transform_list)


class DensePoseDataset(DensePoseBaseDataset):
	
	def __init__(self, opt, is_for_train):
		super(DensePoseDataset, self).__init__(opt, is_for_train)
		self._name = 'DensePoseDataset'

	def _load_input(self, vid_info):
		num_frames = vid_info['num_frames']
		num_cams = vid_info['num_cams']

		# Select frame and camera ids:
		while True:
			frame_id = np.random.randint(0, num_frames)
			cam_id = np.random.randint(0, num_cams)

			# UV:
			uv_path = vid_info['uvs'][cam_id][frame_id]
			data = load_pickle_file(uv_path)
			if data['bbox'] is None:
				continue
			else:
				break

		uv_raw = decode_uv(data['uv'])
		mask = uv_raw[:,:,0]
		h,w = mask.shape[:2]
		uvs = []
		for i in range(1,25):
			x, y = np.where(mask[:, :] == i)
			uv = np.zeros((h,w,2), dtype=np.float32)
			uv[x, y, 0] = uv_raw[x, y, 1] / 255.0
			uv[x, y, 1] = uv_raw[x, y, 2] / 255.0
			uvs.append(np.expand_dims(uv,0))
		uvs = np.concatenate(uvs, axis=0)
		uvs = np.transpose(uvs, (0,3,1,2))
		assert uvs.shape[0] == 24

		# SMPL:
		s1 = vid_info['smpl'][cam_id]['cams'][frame_id]
		s2 = vid_info['smpl'][cam_id]['pose'][frame_id]
		s3 = vid_info['smpl'][cam_id]['shape'][frame_id]
		smpl = np.concatenate((s1,s2,s3),axis=0)
		# smpl = np.expand_dims(smpl, axis=0)

		# Image:
		image_path = vid_info['images'][cam_id][frame_id]
		image = cv_utils.read_cv2_img(image_path)

		return image, smpl, uvs, mask