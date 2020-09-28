# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tensorboardX import SummaryWriter

from .network import Segmentor
from .logger import log
from .evaluation import Evaluator
from .datasets import *


#  Potential fix for 'RuntimeError: received 0 items of ancdata'
#  https://github.com/pytorch/pytorch/issues/973
#  duplicate in data_manager
torch.multiprocessing.set_sharing_strategy('file_system')


class Trainer:
    """"""
    def __init__(self, options):

        print('setting up...')
        self.opt = options

        # Parse config file
        self.config = load_config(self.opt.config_path)

        # Create network and load if required
        self.model = Segmentor(pretrained=True, use_PSP=not self.opt.no_PSP)
        if self.opt.load_path is not None:
            self.load_model()
        if torch.cuda.is_available():
            self.model.cuda()
        self.sigmoid = nn.Sigmoid()

        # Set up losses and optimiser
        self.evaluator = Evaluator()
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=10)
        print('models done!')

        # Create dataloaders
        self.train_loader, self.val_loader = self.create_dataloaders()
        self.val_iter = iter(self.val_loader)
        print('datasets done!')
        print('training images: {}'.format(len(self.train_loader.dataset)))
        print('validation images: {}'.format(len(self.val_loader.dataset)))

        # Set up tensorboard writers
        self.train_writer = SummaryWriter(os.path.join(self.opt.log_path,
                                                       self.opt.model_name, 'train'))
        self.val_writer = SummaryWriter(os.path.join(self.opt.log_path,
                                                     self.opt.model_name, 'val'))

    def create_dataloaders(self):
        train_datasets = []
        val_datasets = []

        for dataset in self.opt.training_datasets:
            dataset_path = self.config[dataset]['dataset']
            textfile_path = self.config[dataset]['training_data']
            train_textfile = os.path.join('splits', dataset, 'train.txt')
            val_textfile = os.path.join('splits', dataset, 'val.txt')
            dataset_class = get_dataset_class(dataset)

            with open(train_textfile) as file:
                train_files = file.read().splitlines()
            with open(val_textfile) as file:
                val_files = file.read().splitlines()

            if dataset == 'matterport':
                train_files = train_files[:5000]

            train_datasets.append(dataset_class(dataset_path,
                                                train_files,
                                                self.opt.height, self.opt.width,
                                                is_train=True))

            val_datasets.append(dataset_class(dataset_path,
                                              val_files,
                                              self.opt.height, self.opt.width,
                                              is_train=False))

        train_datasets = ConcatDataset(train_datasets)
        train_loader = DataLoader(train_datasets, batch_size=self.opt.batch_size, shuffle=True,
                                  num_workers=self.opt.num_workers)

        val_datasets = ConcatDataset(val_datasets)
        val_loader = DataLoader(val_datasets, batch_size=self.opt.batch_size, shuffle=True,
                                num_workers=min(2, self.opt.num_workers))

        return train_loader, val_loader

    def train(self):
        print('training')
        self.step = 0
        for self.epoch in range(self.opt.epochs):
            self.run_epoch()

    def run_epoch(self):

        self.scheduler.step()

        for i, inputs in enumerate(self.train_loader):

            # Run through network computing losses
            predictions, batch_loss = self.forward(inputs)

            # only log full scale outputs
            output_to_log = {'ground': self.sigmoid(predictions[('ground', 3)])}

            # Update weights
            self.model.zero_grad()
            batch_loss.backward()
            self.optimiser.step()
            self.lr = self.scheduler.get_lr()[0]

            # Log to tensorboard and run validation
            if (self.step % self.opt.log_freq) == 0:
                # get losses and reset tracked losses
                tracked_losses = self.evaluator.get_tracked_losses()

                log(self.train_writer, inputs, output_to_log, tracked_losses, self.lr,
                    step=self.step, num_outputs=10)

                train_loss = tracked_losses['loss']
                val_losses = self.run_validation()
                val_loss = val_losses['loss']

                print("Epoch {} -- Step {} -- Train Loss {} -- Val Loss {}".format(self.epoch,
                                                                                   self.step,
                                                                                   train_loss,
                                                                                   val_loss))

            self.step += 1
        # End of epoch - save model
        self.save_model()

    def run_validation(self, batches=10, run_logger=True):

        with torch.no_grad():
            for _ in range(batches):
                try:
                    inputs = self.val_iter.next()
                except StopIteration:
                    self.val_iter = iter(self.val_loader)
                    inputs = self.val_iter.next()

                predictions, _ = self.forward(inputs)

            tracked_losses = self.evaluator.get_tracked_losses()

            if run_logger:

                # only log full scale outputs
                output_to_log = {'ground': self.sigmoid(predictions[('ground', 3)])}

                log(self.val_writer, inputs, output_to_log, tracked_losses, self.lr, step=self.step,
                    num_outputs=10)

            return tracked_losses

    def forward(self, inputs):
        image = inputs['image'].requires_grad_().float()
        ground_mask = inputs['ground_mask']
        loss_mask = inputs['labelled_pix']

        if torch.cuda.is_available():
            image = image.cuda()
            loss_mask = loss_mask.cuda()
            ground_mask = ground_mask.cuda()

        output = self.model(image)

        # Upsize predictions at different scales and save to dictionary
        predictions = {}
        for scale, out in enumerate(output):
            out = nn.functional.interpolate(out, size=(self.opt.height, self.opt.width),
                                            mode='bilinear', align_corners=False)
            out = out.squeeze(1)  # remove channels dimension
            predictions[('ground', scale)] = out

        # compute losses
        batch_loss = self.evaluator.compute_losses(predictions, ground_mask, loss_mask)

        return predictions, batch_loss

    def save_model(self):
        save_path = os.path.join(self.opt.log_path, self.opt.model_name, 'models')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(self.model.state_dict(), os.path.join(save_path,
                                                         'epoch_{}.pth'.format(self.epoch)))

    def load_model(self):
        """ Load a pretrained model """

        print('loading weights from {}...'.format(self.opt.load_path))
        weights = torch.load(self.opt.load_path)
        if torch.cuda.is_available():
            self.model.load_state_dict(weights)
        else:
            self.model.load_state_dict(weights, map_location='cpu')
        print('successfully loaded weights!')


def get_dataset_class(dataset_type):
    return {
        'ADE20K': ADE20KDataset,
        'cityscapes': CityscapesDataset,
        'matterport': MatterportDataset
    }[dataset_type]


def load_config(config):
    with open(config, 'r') as fh:
        config = yaml.safe_load(fh)
    return config
