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


class MatterportDataset(BaseDataset):

    ground_labels = np.array([
        1  # ground

    ]).astype(float)

    def __init__(self, datapath, filenames, height, width, is_train=False, has_gt=True):
        """Pytroch dataset for Matterport dataset"""
        super().__init__(datapath, filenames, height, width, is_train, has_gt=has_gt)

    def _load_image(self, index):
        scan, pos, height, direction = self.filenames[index].split(' ')
        return self.pil_loader(os.path.join(self.datapath, 'sample_dataset/v1/scans', scan, scan,
                                            'matterport_color_images',
                                            '{}_i{}_{}.jpg'.format(pos, height, direction)))

    def _load_annotation(self, index):
        scan, pos, height, direction = self.filenames[index].split(' ')

        labels = np.load(os.path.join(self.datapath, 'sample_dataset/v1/scans', scan,
                                      'nia_ground_masks',
                                      'out_{}_{}_{}_visibleground.npy'.format(pos,
                                                                              height,
                                                                              direction)))
        # in keeping with other datasets, return a PIL Image
        return Image.fromarray((labels > 0).astype(np.uint8))

    def _preprocess(self, image, labels):
        # make the images smaller for more context
        if self.is_train:
            width, height = image.size
            resize_factor = 0.25 + 0.75 * random.random()
            image = image.resize((int(width * resize_factor), int(height * resize_factor)),
                                 resample=Image.LANCZOS)
            labels = labels.resize((int(width * resize_factor), int(height * resize_factor)),
                                   resample=Image.NEAREST)
        return image, labels

    def _process_labels(self, labels):
        labels = np.array(labels)

        # generate binary mask
        ground_mask = self.generate_mask(labels)
        ground_mask = torch.from_numpy(ground_mask).float()
        # get a dummy ignore mask so datasets all return the same keys
        labelled_pix = torch.ones_like(ground_mask).float()

        return ground_mask, labelled_pix
