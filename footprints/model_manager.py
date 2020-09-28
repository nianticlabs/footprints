# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os

import torch

from .network import FootprintNetwork


class ModelManager:

    def __init__(self, save_folder=None, use_cuda=True, is_inference=False,
                 learning_rate=1e-4, lr_step_size=10):

        self.save_folder = save_folder
        self.use_cuda = use_cuda

        self.model = self.model = FootprintNetwork(pretrained=True)
        if self.use_cuda:
            self.model.cuda()

        if not is_inference:
            self.optimiser = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=lr_step_size)

    def load_model(self, weights_path, load_optimiser=False):

        print('loading model weights from {}...'.format(weights_path))
        weights = torch.load(os.path.join(weights_path, 'model.pth'))
        if torch.cuda.is_available():
            self.model.load_state_dict(weights)
        else:
            self.model.load_state_dict(weights, map_location='cpu')
        print('successfully loaded weights!')

        if load_optimiser:
            print('loading optimiser...')
            weights = torch.load(os.path.join(weights_path, 'optimiser.pth'))
            self.optimiser.load_state_dict(weights)
            print('successfully loaded optimiser!')

    def save_model(self, folder_name):

        save_path = os.path.join(self.save_folder, folder_name)
        print('saving weights to {}...'.format(save_path))
        os.makedirs(save_path, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(save_path,
                                                         'model.pth'))
        torch.save(self.optimiser.state_dict(), os.path.join(save_path,
                                                             'optimiser.pth'))
        print('success!')
