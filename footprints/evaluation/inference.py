# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import tqdm

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from .datasets import InferenceDataset
from ..model_manager import ModelManager
from ..utils import load_config
from ..datasets import get_inference_dataset_class

COLORMAP = plt.get_cmap('plasma', 256)  # for plotting


class InferenceManager:

    def __init__(self, options):
        print('---------------')
        print('setting up...')
        self.opt = options

        # Parse config file
        self.config = load_config(self.opt.config_path)

        if self.opt.inference_save_path is None:
            self.savepath = os.path.join(self.opt.load_path,
                                         '{}_predictions'.format(self.opt.inference_data_type))
        else:
            self.savepath = self.opt.inference_save_path

        print('saving output to {}'.format(self.savepath))

        # Create network and optimiser
        self.model_manager = ModelManager(use_cuda=torch.cuda.is_available(),
                                          is_inference=True)
        self.model_manager.load_model(weights_path=self.opt.load_path, load_optimiser=False)

        # extract model, optimiser and scheduler for easier access
        self.model = self.model_manager.model
        self.model.eval()
        print('models done!')

        self.loader, self.dataset = self.create_dataloaders()
        self.sigmoid = nn.Sigmoid()

        print('inference setup complete!')
        print('---------------')

    def create_dataloaders(self):
        dataset = self.opt.inference_data_type
        raw_data_path = self.config[dataset]['dataset']
        textfile_path = os.path.join('splits', dataset, 'test.txt')
        dataset_class = get_inference_dataset_class(dataset)

        with open(textfile_path) as file:
            test_files = file.read().splitlines()

        dataset = dataset_class(raw_data_path,
                                test_files,
                                self.opt.height, self.opt.width,
                                )

        loader = DataLoader(dataset, batch_size=self.opt.batch_size, shuffle=False,
                            num_workers=self.opt.num_workers)

        return loader, dataset

    def run(self):
        """ Run an image or images through the network """

        print('running inference...')

        with torch.no_grad():
            for inputs in tqdm.tqdm(self.loader):
                preds, visualisations = self.test_batch(inputs)

                for i, pred in enumerate(preds):
                    idx = inputs['idx'][i]
                    viz = visualisations[i] if self.opt.save_test_visualisations else None
                    self.dataset.save_result(idx, pred, self.savepath, viz)

        print('finished testing!')

    def test_batch(self, inputs):

        if torch.cuda.is_available():
            inputs['image'] = inputs['image'].cuda()

        # just take max resolution prediction
        preds = self.model(inputs['image'])['1/1']

        # apply sigmoid to mask channels - not applied in network for stability of BCE loss
        preds[:, 0:2] = self.sigmoid(preds[:, 0:2])
        preds = preds.cpu().numpy()

        # prepare visualisation
        visualisations = []
        if self.opt.save_test_visualisations:
            # save visualisation of all ground channel
            for j, image in enumerate(inputs['image']):
                image = image.float().detach().cpu().numpy().transpose([1, 2, 0])
                pred = (preds[j, 1] > 0.5).astype(float)
                pred_cm = COLORMAP(pred)[..., :3]
                viz = np.concatenate([image, pred_cm], 1)
                visualisations.append(viz)

        return preds, visualisations
