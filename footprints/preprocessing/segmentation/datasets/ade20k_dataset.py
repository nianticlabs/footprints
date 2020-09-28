# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import numpy as np
import torch

from .base_dataset import BaseDataset


class ADE20KDataset(BaseDataset):

    ground_labels = np.array([
        976,  # floor
        2131,  # road
        1125,  # grass
        2377,  # pavement
        838,  # ground
        913,  # field
        2212,  # sand
        1788,  # path
        2530,  # stairs
        2185,  # runway
        2531,  # staircase
        738,  # track
        1401,  # soil
        1494,  # manhole
    ]).astype(float)

    def __init__(self, datapath, filenames, height, width, is_train=False, has_gt=True):
        """Pytroch dataset for ade20k dataset
        Here we are extracting ground pixels only and treating as a binary classification problem"""
        super().__init__(datapath, filenames, height, width, is_train, has_gt=has_gt)

    def _load_image(self, index):
        filename = os.path.splitext(self.filenames[index])[0]
        return self.pil_loader(os.path.join(self.datapath, filename + '.jpg'))

    def _load_annotation(self, index):
        filename = os.path.splitext(self.filenames[index])[0]
        return self.pil_loader(os.path.join(self.datapath, filename + '_seg.png'))

    def _preprocess(self, image, labels):
        return image, labels

    def _process_labels(self, labels):
        labels = np.array(labels)
        labels = labels[..., 0] // 10 * 256 + labels[..., 1]  # ID = R / 10 * 256 + G

        # generate binary mask
        ground_mask = self._generate_mask(labels)
        ground_mask = torch.from_numpy(ground_mask).float()
        # get a dummy ignore mask so datasets all return the same keys
        labelled_pix = torch.ones_like(ground_mask).float()

        return ground_mask, labelled_pix
