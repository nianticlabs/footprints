# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os

import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from .dataset_utils import *


class InferenceDataset(data.Dataset):

    def __init__(self, data_path, filenames, height, width):
        super(InferenceDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width

        self.resizer = transforms.Resize((height, width), interpolation=Image.ANTIALIAS)
        self.totensor = transforms.ToTensor()
        self.pil_loader = pil_loader

    def __len__(self):
        return len(self.filenames)

    def _load_image(self, index):
        raise NotImplementedError

    def save_result(self, savepath, filename, prediction, visualisation=None):

        _savepath = os.path.join(savepath, 'data')
        os.makedirs(_savepath, exist_ok=True)
        np.save(os.path.join(_savepath, '{}.npy'.format(str(filename).zfill(10))),
                prediction.astype(np.float16))

        if visualisation is not None:
            _savepath = os.path.join(savepath, 'visualisations')
            os.makedirs(_savepath, exist_ok=True)
            plt.imsave(os.path.join(_savepath, '{}.jpg'.format(str(filename).zfill(10))),
                       visualisation)

    def __getitem__(self, index):

        image = self._load_image(index)
        image = self.resizer(image)
        image = self.totensor(image)

        inputs = {'image': image, 'idx': index}

        return inputs


class KITTIInferenceDataset(InferenceDataset):

    def __init__(self, data_path, filenames, height, width,
                 image_ext='jpg'):
        super(KITTIInferenceDataset, self).__init__(data_path, filenames, height, width)

        self.image_ext = image_ext

    def _parse_index(self, index):
        seq, frame, side = self.filenames[index].split(' ')
        if side == 'l':
            side = 'image_02'
        else:
            side = 'image_03'

        return seq, frame, side

    def _load_image(self, index):

        seq, frame, side = self._parse_index(index)

        image = self.pil_loader(os.path.join(self.data_path, seq, side, 'data',
                                             '{}.{}'.format(str(frame).zfill(10), self.image_ext)))

        return image

    def save_result(self, index, prediction, savepath, visualisation=None):

        seq, frame, side = self._parse_index(index)

        savepath = os.path.join(savepath, seq, side)

        super(KITTIInferenceDataset, self).save_result(savepath, frame, prediction, visualisation)


class MatterportInferenceDataset(InferenceDataset):

    def __init__(self, data_path, filenames, height, width):
        super(MatterportInferenceDataset, self).__init__(data_path, filenames, height, width)

        self.image_ext = image_ext

    def _load_image(self, index):

        scan, pos, height, direction = self.filenames[index].split(' ')

        image = self.pil_loader(os.path.join(self.datapath, 'sample_dataset/v1/scans', scan, scan,
                                        'matterport_color_images',
                                        '{}_i{}_{}.jpg'.format(pos, height, direction)))

        return image

    def save_result(self, index, prediction, savepath, visualisation=None):

        scan, pos, height, direction = self.filenames[index].split(' ')

        savepath = os.path.join(savepath, scan)
        filemame = '{}_{}_{}'.format(pos, height, direction)

        super(MatterportInferenceDataset, self).save_result(savepath, filemame,
                                                            prediction, visualisation)
