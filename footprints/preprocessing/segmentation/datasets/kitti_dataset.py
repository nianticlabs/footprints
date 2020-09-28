# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from .dataset_utils import *


class KITTIDataset:

    ground_labels = np.array([6,  # ground
                              7,  # road
                              8,  # pavement
                              9,  # parking
                              22  # terrain
    ]).astype(float)

    def __init__(self, datapath, filenames, height, width, is_train=False):
        """Pytroch dataset for mscoco dataset
        Here we are extracting ground pixels only and treating as a binary classification problem"""
        super().__init__()
        self.datapath = datapath
        self.filenames = filenames
        self.height = height
        self.width = width

        self.resizer = {'image': transforms.Resize((height, width), interpolation=Image.ANTIALIAS),
                        'labels': transforms.Resize((height, width), interpolation=Image.NEAREST)}

        self.is_train = is_train

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """
        Load an image along with its semantic labelling - here binary.
        We randomly crop our image to the desired aspect ratio, before resizing
        """
        inputs = {}

        frame, _ = self.filenames[index].split()

        # Load image and mask
        image = pil_loader(os.path.join(self.datapath, frame))

        frame = frame.replace('image_2', 'semantic')
        labels = pil_loader(os.path.join(self.datapath, frame))

        # Resize
        image = self.resizer['image'](image)
        labels = self.resizer['labels'](labels)

        to_tensor = transforms.ToTensor()
        image = to_tensor(image).float()
        labels = np.array(labels)[..., 0]  # all channels are equal

        # labels to segmentation mask
        ground_mask = self.generate_mask(labels)
        ground_mask = torch.from_numpy(ground_mask).float()

        # if we are using coarse labels, then unlabelled regions have 0 id so we need a mask
        labelled_pix = torch.from_numpy((labels != 0).astype(float)).float()

        inputs['image'] = image
        inputs['ground_mask'] = ground_mask
        inputs['labelled_pix'] = labelled_pix

        return inputs

    def generate_mask(self, labels):
        """create ground mask and moving object mask"""
        ground_mask = np.in1d(labels, self.ground_labels).reshape(labels.shape).astype(float)
        return ground_mask