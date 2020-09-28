# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os

import numpy as np
import tqdm

import torch
from torch.utils.data import DataLoader

from .network import Segmentor
from .logger import COLORMAP
from .datasets import *
from .train import load_config


class Tester:
    """"""
    def __init__(self, options):

        print('setting up...')
        self.opt = options

        # Parse config file
        self.config = load_config(self.opt.config_path)
        self.path_data = self.config[self.opt.test_data_type]
        self.save_path = os.path.join(self.path_data['training_data'], self.opt.test_save_folder)

        # Create network
        self.model = Segmentor(pretrained=False, use_PSP=not self.opt.no_PSP)
        self.load_model()
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
        self.sigmoid = torch.nn.Sigmoid()

        # Concatenate train files and val files
        filenames = []
        for textfile in ['train.txt', 'val.txt']:
            full_path = os.path.join('splits', self.opt.test_data_type, textfile)
            with open(full_path) as file:
                filenames += file.read().splitlines()
        filenames = sorted(filenames)

        # Create dataloader
        self.dataset = get_dataset_class(self.opt.test_data_type)(self.path_data['dataset'],
                                                                  filenames,
                                                                  self.opt.height, self.opt.width,
                                                                  )
        self.loader = DataLoader(self.dataset, batch_size=self.opt.batch_size, shuffle=False,
                                 num_workers=self.opt.num_workers)

    def test(self):
        """ Run an image or images through the network """

        print('running inference...')

        with torch.no_grad():
            for inputs in tqdm.tqdm(self.loader):
                preds, visualisations = self.test_batch(inputs)

                for i, pred in enumerate(preds):
                    idx = inputs['idx'][i]
                    viz = visualisations[i] if self.opt.save_test_visualisations else None
                    self.dataset.save_result(idx, pred, self.save_path, viz)

        print('finished testing!')

    def test_batch(self, inputs):

        if torch.cuda.is_available():
            inputs['image'] = inputs['image'].cuda()

        preds = self.model(inputs['image'])
        # just take max resolution prediction
        preds = self.sigmoid(preds[3][:, 0:1]).detach().cpu().numpy()

        # prepare visualisation
        visualisations = []
        if self.opt.save_test_visualisations:
            for j, image in enumerate(inputs['image']):
                image = image.float().detach().cpu().numpy().transpose([1, 2, 0])
                pred = preds[j, 0]
                pred_cm = COLORMAP(pred)[..., :3]
                viz = np.concatenate([image, pred_cm], 1)
                visualisations.append(viz)

        return preds, visualisations

    def load_model(self):
        """ Load a pretrained model """

        print('loading weights from {}...'.format(self.opt.load_path))
        if torch.cuda.is_available():
            weights = torch.load(self.opt.load_path)
        else:
            weights = torch.load(self.opt.load_path, map_location='cpu')
        self.model.load_state_dict(weights)
        print('successfully loaded weights!')


def get_dataset_class(dataset_name):
    """
    Helper function which returns class corresponding to a dataset name
    """
    return {
        "kitti": KITTIInferenceDataset,
        "matterport": MatterportInferenceDataset
        }[dataset_name]