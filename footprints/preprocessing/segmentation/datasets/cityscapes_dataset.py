# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import numpy as np
import random
from PIL import Image
import torch

from .base_dataset import BaseDataset


class CityscapesDataset(BaseDataset):

    ground_labels = np.array([6,  # ground
                              7,  # road
                              8,  # pavement
                              9,  # parking
                              22  # terrain
    ]).astype(float)

    def __init__(self, datapath, filenames, height, width, is_train=False, has_gt=True):
        """Pytroch dataset for cityscapes dataset
        Here we are extracting ground pixels only and treating as a binary classification problem"""
        super().__init__(datapath, filenames, height, width, is_train, has_gt=has_gt)

    def _load_image(self, index):
        folder, city, frame = self.filenames[index].split()
        return self.pil_loader(os.path.join(self.datapath, 'leftImg8bit', folder, city,
                               frame + '_leftImg8bit.png'))

    def _load_annotation(self, index):
        folder, city, frame = self.filenames[index].split()
        # try to load fine ground truth, else load coarse
        try:
            labels = self.pil_loader(os.path.join(self.datapath, 'gtFine', folder,
                                                  city, frame + '_gtFine_labelIds.png'))
        except FileNotFoundError:
            labels = self.pil_loader(os.path.join(self.datapath, 'gtCoarse', folder + '_extra',
                                                  city, frame + '_gtCoarse_labelIds.png'))
        return labels

    def _preprocess(self, image, labels):
        # crop out ego car,
        image = image.crop((0, 0, image.size[0], 795))
        labels = labels.crop((0, 0, image.size[0], 795))

        # make the images smaller for more context
        if self.is_train:
            width, height = image.size
            resize_factor = 0.4 + 0.6 * random.random()
            image = image.resize((int(width * resize_factor), int(height * resize_factor)),
                                 resample=Image.LANCZOS)
            labels = labels.resize((int(width * resize_factor), int(height * resize_factor)),
                                   resample=Image.NEAREST)
        return image, labels

    def _process_labels(self, labels):
        labels = np.array(labels)[..., 0]  # all channels are equal

        # labels to segmentation mask
        ground_mask = self._generate_mask(labels)
        ground_mask = torch.from_numpy(ground_mask).float()

        # if we are using coarse labels, then unlabelled regions have 0 id so we need a mask
        labelled_pix = torch.from_numpy((labels != 0).astype(float)).float()

        return ground_mask, labelled_pix
