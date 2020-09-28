# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

from .dataset_utils import *


class BaseDataset(data.Dataset):

    # augmentations
    max_angle = 5
    brightness = 0.3
    contrast = 0.2
    saturation = 0.3
    hue = 0.1

    ground_labels = None

    def __init__(self, datapath, filenames, height, width, is_train=False, has_gt=True):
        """Abstract class for ground segmentation datasets"""
        super().__init__()
        self.datapath = datapath
        self.filenames = filenames
        self.height = height
        self.width = width
        self.is_train = is_train
        self.has_gt = has_gt
        self.color_aug = transforms.ColorJitter(self.brightness, self.contrast,
                                                self.saturation, self.hue)
        self.to_tensor = transforms.ToTensor()
        self.pil_loader = pil_loader

    def __len__(self):
        return len(self.filenames)

    def _load_image(self, index):
        raise NotImplementedError

    def _load_annotation(self, index):
        raise NotImplementedError

    def _preprocess(self, image, labels):
        raise NotImplementedError

    def _process_labels(self, labels):
        raise NotImplementedError

    def _generate_mask(self, labels):
        """create ground mask and moving object mask"""
        ground_mask = np.in1d(labels, self.ground_labels).reshape(labels.shape).astype(float)
        return ground_mask

    def _augment_data(self, image, labels):
        """ Adjust color and apply random flipping
        """
        rands = torch.rand(2)
        # Color augmentation
        if rands[0] > 0.5:
            image = self.color_aug(image)
        # Flipping
        if rands[1] > 0.5:
            image = F.hflip(image)
            labels = F.hflip(labels)

        return image, labels

    def __getitem__(self, index):

        inputs = {}
        image = self._load_image(index)

        if self.has_gt:
            labels = self._load_annotation(index)
        else:
            labels = Image.from_array(np.zeros_like(np.array(image)))

        image, labels = self._preprocess(image, labels)

        # Crop and apply augmentations
        image, labels = prepare_size(image, labels, self.height, self.width,
                                     keep_aspect_ratio=True)
        if self.is_train:
            image, labels = self._augment_data(image, labels)

        # convert to tensors and save in dictionary
        image = self.to_tensor(image).float()
        inputs['image'] = image

        ground_mask, labelled_pix = self._process_labels(labels)
        inputs['ground_mask'] = ground_mask
        inputs['labelled_pix'] = labelled_pix

        return inputs

