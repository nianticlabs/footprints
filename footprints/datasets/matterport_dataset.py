# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os

import numpy as np
import cv2
import random

from PIL import Image

from .footprint_dataset import FootprintsDataset


class MatterportDataset(FootprintsDataset):

    depth_scaling = 0.00025  # Convert to real world depths (metres) - given by Matterport

    def __init__(self, raw_data_path, training_data_path, filenames, height, width, no_depth_mask,
                 is_train=False, **kwargs):
        """ Matterport dataset """
        super().__init__(raw_data_path, training_data_path, filenames, height, width, is_train)

        self.no_depth_mask = no_depth_mask
        self.process_depth_mask = True
        self.footprint_threshold = 0.75

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """
        Load an image, its visible footprint, hidden geometry and depth.
        """
        inputs = {}

        do_flip = True if self.is_train and random.random() > 0.5 else False
        color_aug = True if self.is_train and random.random() > 0.5 else False

        scan, pos, height, direction = self.filenames[index].split(' ')

        image_path = os.path.join(self.raw_data_path, scan,
                                  scan, 'matterport_color_images',
                                  '{}_i{}_{}.jpg'.format(pos, height, direction))
        image = self.load_and_resize_image(image_path, do_flip)

        depth_path = os.path.join(self.raw_data_path, scan,
                                  scan, 'matterport_depth_images',
                                  '{}_d{}_{}.png'.format(pos, height, direction))
        depth = np.array(self.load_and_resize_image(depth_path, do_flip, method=Image.NEAREST))

        visible_ground_path = os.path.join(self.training_data_path, 'ground_seg', scan,
                                           'data', '{}_{}_{}.npy'.format(pos, height, direction))
        visible_ground = self.load_and_resize_npy(visible_ground_path, do_flip,
                                                  method=cv2.INTER_AREA)
        visible_ground = visible_ground > self.footprint_threshold

        ground_depth_path = os.path.join(self.training_data_path, 'hidden_depth', scan, 'data',
                                         '{}_{}_{}.npy'.format(pos, height, direction))
        ground_depth = self.load_and_resize_npy(ground_depth_path, do_flip, method=cv2.INTER_AREA)

        # depths in 16bit PNG -> scale to metric
        depth = depth.astype(float) * self.depth_scaling

        # missing pixels have depth 0.1 not 0
        ground_depth[ground_depth == 0.1] = 0

        # limit hidden ground distance
        ground_depth *= (ground_depth < 10.0)

        # matterport has no moving objects
        moving_objects = np.zeros_like(depth)

        # load not ground - may not exist
        try:
            depth_mask_path = os.path.join(self.training_data_path, 'depth_masks',
                                           scan, 'data',
                                           '{}_{}_{}.npy'.format(pos, height, direction))
            depth_mask = self.load_and_resize_npy(depth_mask_path, do_flip)

            # reduce size of not ground
            if self.process_depth_mask:
                depth_mask = self.filter_depth_mask(depth_mask)

        except FileNotFoundError:
            depth_mask = np.zeros_like(depth)

        if self.no_depth_mask:
            depth_mask *= 0

        # set definitely not ground pixels to 0
        ground_depth[depth_mask.astype(bool)] = 0

        inputs['image'] = image
        inputs['visible_ground'] = visible_ground
        inputs['depth'] = depth
        inputs['ground_depth'] = ground_depth
        inputs['moving_object_mask'] = moving_objects
        inputs['depth_mask'] = depth_mask

        # augment data and convert to tensors
        inputs = self.preprocess(inputs, color_aug=color_aug)

        return inputs
