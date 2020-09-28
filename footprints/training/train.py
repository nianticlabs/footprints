# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import time
import numpy as np
import random

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from loguru import logger

from ..datasets import get_dataset_class
from .evaluation import Evaluator
from ..model_manager import ModelManager
from .logger import log, TimeLogger
from ..utils import sec_to_hm_str, load_config


#  Potential fix for 'RuntimeError: received 0 items of ancdata'
#  https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')


SEED = 10

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)


class TrainManager:
    """
    Main training script called from main.py.

    """

    def __init__(self, options):
        logger.info('---------------')
        logger.info('setting up...')
        self.opt = options

        # Parse config file
        self.config = load_config(self.opt.config_path)

        # Create network and optimiser
        self.model_manager = ModelManager(save_folder=os.path.join(self.opt.log_path,
                                                                   self.opt.model_name,
                                                                   'models'),
                                          use_cuda=torch.cuda.is_available(),
                                          learning_rate=self.opt.lr,
                                          lr_step_size=10)
        if self.opt.load_path is not None:
            self.model_manager.load_model(weights_path=self.opt.load_path, load_optimiser=True)

        # extract model, optimiser and scheduler for easier access
        self.model = self.model_manager.model
        self.optimiser = self.model_manager.optimiser
        self.scheduler = self.model_manager.scheduler
        logger.info('models done!')

        # Create datamanager to generate batches etc
        self.train_loader, self.val_loader = self.create_dataloaders()
        self.val_iter = iter(self.val_loader)
        logger.info('datasets done!')
        logger.info('dataset info:')
        logger.info('train size - {} images'.format(len(self.train_loader.dataset)))
        logger.info('validation size - {} images'.format(len(self.val_loader.dataset)))

        # Set up losses
        self.evaluator = Evaluator(depth_range=self.opt.depth_range,
                                   footprint_prior=self.opt.footprint_prior)

        # Set up tensorboard writers and logger
        self.train_writer = SummaryWriter(os.path.join(self.opt.log_path,
                                                       self.opt.model_name, 'train'))
        self.val_writer = SummaryWriter(os.path.join(self.opt.log_path,
                                                     self.opt.model_name, 'val'))
        os.makedirs(self.opt.log_path, exist_ok=True)
        self.timer = TimeLogger()

        # self.logger = Logger()
        self.step = 0
        self.num_total_steps = len(self.train_loader) * self.opt.epochs

        logger.info('training setup complete!')
        logger.info('---------------')

    def create_dataloaders(self):
        dataset = self.opt.training_dataset
        raw_data_path = self.config[dataset]['dataset']
        training_data_path = self.config[dataset]['training_data']
        train_textfile = os.path.join('splits', dataset, 'train.txt')
        val_textfile = os.path.join('splits', dataset, 'val.txt')
        dataset_class = get_dataset_class(dataset)

        with open(train_textfile) as file:
            train_files = file.read().splitlines()
        with open(val_textfile) as file:
            val_files = file.read().splitlines()

        train_dataset = dataset_class(raw_data_path, training_data_path,
                                      train_files,
                                      self.opt.height, self.opt.width,
                                      no_depth_mask=self.opt.no_depth_mask,
                                      moving_objects_method=self.opt.moving_objects_method,
                                      project_down_baseline=self.opt.project_down_baseline,
                                      is_train=True)

        val_dataset = dataset_class(raw_data_path, training_data_path,
                                    val_files,
                                    self.opt.height, self.opt.width,
                                    no_depth_mask=self.opt.no_depth_mask,
                                    moving_objects_method=self.opt.moving_objects_method,
                                    project_down_baseline=self.opt.project_down_baseline,
                                    is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=self.opt.batch_size, shuffle=True,
                                  num_workers=self.opt.num_workers)

        val_loader = DataLoader(val_dataset, batch_size=self.opt.batch_size, shuffle=True,
                                num_workers=min(2, self.opt.num_workers))

        return train_loader, val_loader

    def train(self):

        logger.info('training...')
        self.start_time = time.time()
        for self.epoch in range(self.opt.epochs):
            self.run_epoch()

        logger.info('training complete!')

    def run_epoch(self):

        for batch_idx, inputs in enumerate(self.train_loader):

            before_time = time.time()
            outputs, losses = self.process_batch(inputs, mode='train', return_batch_loss=True)

            # Update weights
            batch_loss = losses['loss']
            self.model.zero_grad()
            batch_loss.backward()
            self.optimiser.step()
            self.lr = self.scheduler.get_lr()[0]

            self.timer.add_time(timer='train_network_time', time=time.time() - before_time)

            if self.step % 100 == 0:
                losses = self.evaluator.get_averaged_losses(mode='train', reset=False)
                # log to console
                logger.info('Epoch {} -- Batch {} -- Loss {}'.format(self.epoch, batch_idx,
                                                                     losses['loss']))
                # show times
                self.timer.print_time()

                time_sofar = time.time() - self.start_time
                training_time_left = (self.num_total_steps / self.step - 1.0) \
                         * time_sofar if self.step != 0 else 0
                logger.info("time elapsed/left: {}/{}"
                            .format(sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

                # validate and log
                if self.step % self.opt.log_freq == 0:
                    losses = self.evaluator.get_averaged_losses(mode='train', reset=True)

                    before_time = time.time()
                    log(self.train_writer, inputs, outputs, losses, self.lr, self.step)
                    self.timer.add_time(timer='log_time', time=time.time() - before_time)

                    self.model.eval()
                    self.val()
                    self.model.train()

            self.step += 1

        logger.info('Epoch {} complete!'.format(self.epoch))
        self.model_manager.save_model(folder_name='weights_{}'.format(self.epoch))
        self.scheduler.step()

    def val(self):

        before_time = time.time()
        logger.info('validating...')
        with torch.no_grad():
            for _ in range(self.opt.val_batches):
                try:
                    inputs = self.val_iter.next()
                except StopIteration:
                    self.val_iter = iter(self.val_loader)
                    inputs = self.val_iter.next()

                outputs, _ = self.process_batch(inputs, mode='val', return_batch_loss=False)

        logger.info('validation complete!')

        losses = self.evaluator.get_averaged_losses(mode='val', reset=True)

        self.timer.add_time(timer='val_time', time=time.time() - before_time)

        before_time = time.time()
        log(self.val_writer, inputs, outputs, losses, self.lr, self.step)
        self.timer.add_time(timer='log_time', time=time.time() - before_time)

    def process_batch(self, inputs, mode='train', return_batch_loss=False):

        # move to GPU
        if torch.cuda.is_available():
            for key, inp in inputs.items():
                inputs[key] = inp.cuda()

        outputs = self.model(inputs['image'])
        losses = self.evaluator.compute_losses(inputs, outputs, mode=mode,
                                               return_batch_loss=return_batch_loss)
        return outputs, losses
