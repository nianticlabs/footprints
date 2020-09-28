# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os

import numpy as np
import cv2
import random

from .footprint_dataset import FootprintsDataset
from ..utils import pixel_disp_to_depth


class KITTIDataset(FootprintsDataset):

    def __init__(self, raw_data_path, training_data_path, filenames, height, width, no_depth_mask,
                 moving_objects_method, project_down_baseline, is_train=False, **kwargs):
        """ KITTI dataset """
        super().__init__(raw_data_path, training_data_path, filenames, height, width, is_train)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.K[0] *= width
        self.K[1] *= height
        self.baseline = 0.54

        self.footprint_threshold = 0.75
        self.no_depth_mask = no_depth_mask
        self.moving_objects_method = moving_objects_method
        self.project_down_baseline = project_down_baseline
        if self.project_down_baseline:
            assert self.moving_objects_method == 'none', "Error - can't use project down baseline" \
                                                         "with moving object masking"

        self.process_depth_mask = True

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """
        Load an image, its visible footprint, hidden geometry and depth.
        """
        inputs = {}

        seq, frame, side = self.filenames[index].split(' ')
        if side == 'l':
            side = 'image_02'
        else:
            side = 'image_03'
        frame_string = str(frame).zfill(10)

        do_flip = True if self.is_train and random.random() > 0.5 else False
        color_aug = True if self.is_train and random.random() > 0.5 else False

        # Load from disk
        image_path = os.path.join(self.raw_data_path, seq, side, 'data', frame_string + '.jpg')
        image = self.load_and_resize_image(image_path, do_flip)

        visible_ground_path = os.path.join(self.training_data_path, 'ground_seg',
                                               seq, side, 'data', frame_string + '.npy')
        visible_ground = self.load_and_resize_npy(visible_ground_path, do_flip,
                                                  method=cv2.INTER_AREA)
        visible_ground = visible_ground > self.footprint_threshold

        ground_depth_path = os.path.join(self.training_data_path, 'hidden_depths',
                                            seq, side, 'data', frame_string + '.npy')
        ground_depth = self.load_and_resize_npy(ground_depth_path, do_flip, method=cv2.INTER_AREA)

        if self.project_down_baseline:
            ground_depth = np.ones_like(ground_depth)

        try:
            depth_mask_path = os.path.join(self.training_data_path, 'depth_masks',
                                                seq, side, 'data', frame_string + '.npy')
            depth_mask = self.load_and_resize_npy(depth_mask_path, do_flip)

            if self.process_depth_mask:
                depth_mask = self.filter_depth_mask(depth_mask)

        except FileNotFoundError:
            depth_mask = np.zeros_like(ground_depth)

        if self.no_depth_mask:
            depth_mask *= 0

        # set definitely not ground pixels to 0
        ground_depth[depth_mask.astype(bool)] = 0

        disparity_path = os.path.join(self.training_data_path,
                                      'stereo_matching_disps', seq, side, frame_string + '.npy')
        pixel_disparity = \
            self.load_and_resize_npy(disparity_path, do_flip, rescale=True,
                                     method=cv2.INTER_AREA) - 1.25  # see PSM github issue
        depth = pixel_disp_to_depth(pixel_disparity, self.K[0, 0], self.baseline)

        if self.moving_objects_method == 'ours':
            moving_object_path = os.path.join(self.training_data_path, 'moving_objects', seq,
                                              side, 'data', frame_string + '.npy')
            moving_objects = self.load_and_resize_npy(moving_object_path, do_flip)
        else:
            moving_objects = np.zeros((self.height, self.width))

        # moving pixels cant be visible ground
        moving_objects = moving_objects * (1 - visible_ground)
        # moving pixels can't be definitely not ground
        moving_objects = moving_objects * (1 - depth_mask)

        inputs['image'] = image
        inputs['visible_ground'] = visible_ground
        inputs['depth'] = depth
        inputs['ground_depth'] = ground_depth
        inputs['moving_object_mask'] = moving_objects
        inputs['depth_mask'] = depth_mask

        # augment data and convert to tensors
        inputs = self.preprocess(inputs, color_aug=color_aug)

        return inputs
