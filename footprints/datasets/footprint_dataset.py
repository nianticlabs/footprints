# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import numpy as np
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

from skimage.measure import label as find_connected


class FootprintsDataset(data.Dataset):

    def __init__(self, raw_data_path, training_data_path, filenames, height, width, is_train=False):

        """Abstract pytorch dataset for footprints - to be used to learn visible depth, visible
         ground, hidden ground and hidden ground depth.
          """

        self.raw_data_path = raw_data_path
        self.training_data_path = training_data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.is_train = is_train

        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
        self.color_aug = transforms.ColorJitter(self.brightness, self.contrast,
                                                self.saturation, self.hue)

    def __len__(self):
        return len(self.filenames)

    def preprocess(self, inputs, color_aug):
        for key, val in inputs.items():
            if key == 'image':
                if color_aug:
                    val = self.color_aug(val)
                inputs[key] = self.to_tensor(val)
            else:
                inputs[key] = torch.tensor(val).float()

        inputs['all_ground'] = ((inputs['ground_depth'] + inputs['visible_ground']) > 0).float()
        return inputs

    def __getitem__(self, index):
        """
        Load an image, its visible footprint, hidden geometry and depth.
        """
        raise NotImplementedError

    def load_and_resize_image(self, path, do_flip, method=Image.LANCZOS):
        """
        Load image from path with PIL.
        """
        image = Image.open(path).resize(size=(self.width, self.height), resample=method)
        if do_flip:
            image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
        return image

    def load_and_resize_npy(self, path, do_flip, rescale=False, method=cv2.INTER_NEAREST):

        npy = np.load(path).astype(float)
        if len(npy.shape) == 3:
            npy = npy[0]
        if do_flip:
            npy = np.fliplr(npy)

        multiplier = 1.0
        if rescale:
            multiplier = self.width / npy.shape[1]  # for pixel disparity resizing
        npy = cv2.resize(npy, (self.width, self.height), interpolation=method) * multiplier
        return npy

    def filter_depth_mask(self, depth_mask):
        """Remove large connected regions from depth mask"""
        processed = np.zeros_like(depth_mask)
        connected = find_connected(depth_mask)
        for index in range(1, connected.max() + 1):
            size = (connected == index).sum()
            if size < self.width * self.height / 100:
                processed[connected == index] = 1
        depth_mask = processed
        return depth_mask
