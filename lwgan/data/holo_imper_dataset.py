import numpy as np

from .dataset import DatasetBase
from .my_imper_dataset import MyImPerDataset
from .holo_dataset import HoloDataset


class HoloImPerDataset(DatasetBase):

    def __init__(self, opt, is_for_train):
        super(HoloImPerDataset, self).__init__(opt, is_for_train)
        self._name = 'HoloImPerDataset'

        # self.mi = FastLoadMIDataset(opt, is_for_train)
        self.holo = HoloDataset(opt, is_for_train)
        self.imper = MyImPerDataset(opt, is_for_train)

        self._holo_size = len(self.holo)
        self._imper_size = len(self.imper)
        self._dataset_size = self._holo_size + self._imper_size

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, item):
        if item < self._holo_size:
            sample = self.holo[item]
        else:
            sample = self.imper[item]

        return sample

