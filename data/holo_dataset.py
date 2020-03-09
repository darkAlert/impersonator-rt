import os.path
import torchvision.transforms as transforms
from data.dataset import DatasetBase
import numpy as np
from utils import cv_utils
from utils.util import load_pickle_file, ToTensor, ImageTransformer
import glob


__all__ = ['HoloBaseDataset', 'HoloDataset']


class HoloBaseDataset(DatasetBase):

	def __init__(self, opt, is_for_train):
		super(HoloBaseDataset, self).__init__(opt, is_for_train)
		self._name = 'HoloBaseDataset'

		# self._intervals = opt.intervals

		# read dataset
		# self._read_dataset_paths()


class HoloDataset(HoloBaseDataset):
	
	def __init__(self, opt, is_for_train):
		super(HoloDataset, self).__init__(opt, is_for_train)
		self._name = 'HoloDataset'


