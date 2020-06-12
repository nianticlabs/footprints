# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import torch

from .network import FootprintNetwork


class ModelManager:

    def __init__(self, use_cuda, is_train, save_folder=None, learning_rate=None, lr_step_size=None):

        self.model = FootprintNetwork(pretrained=True)
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()

        if is_train:
            self.save_folder = save_folder
            self.optimiser = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=lr_step_size)

    def load_model(self, weights_folder, load_optimiser=False):

        print('Loading model weights from {}'.format(weights_folder))
        weights = torch.load(os.path.join(weights_folder, 'model.pth'))
        if self.use_cuda:
            self.model.load_state_dict(weights)
        else:
            self.model.load_state_dict(weights, map_location='cpu')

        if load_optimiser:
            print('Loading optimiser...')
            weights = torch.load(os.path.join(weights_folder, 'optimiser.pth'))
            self.optimiser.load_state_dict(weights)

    def save_model(self, folder_name):

        save_path = os.path.join(self.save_folder, folder_name)
        print('Saving weights to {}...'.format(save_path))
        os.makedirs(save_path, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(save_path, 'model.pth'))
        torch.save(self.optimiser.state_dict(), os.path.join(save_path, 'optimiser.pth'))
        print('Success!')
