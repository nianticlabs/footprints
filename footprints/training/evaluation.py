# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from collections import defaultdict

import torch

from .losses import LossManager


class Evaluator:
    """
    Class to manage the computation of losses.

    Accumulated losses allow for the averaging of several batch losses to reduce noise.

    """

    def __init__(self, depth_range, footprint_prior):
        self.accumulated_train_losses = defaultdict(list)
        self.accumulated_val_losses = defaultdict(list)

        self.loss_manager = LossManager(depth_range, footprint_prior)

    def compute_losses(self, inputs, outputs, mode='train', return_batch_loss=False):
        """ Compute losses.

         'mode' can be either 'train' or 'val'.

         'losses' will only be added to the respective accumulated losses and not returned by
         default, if the current batch losses are required set 'return_batch_loss' to True"""

        losses = self.loss_manager(predictions=outputs,
                                   targets=inputs)
        for loss_key, loss in losses.items():
            if mode == 'train':
                self.accumulated_train_losses[loss_key].append(loss.detach().cpu())

            elif mode == 'val':
                self.accumulated_val_losses[loss_key].append(loss.detach().cpu())

        if return_batch_loss:
            return losses

    def get_averaged_losses(self, mode, reset=True):

        averaged_losses = {}
        if mode == 'train':
            for loss_key, loss in self.accumulated_train_losses.items():
                loss = torch.stack(loss)
                averaged_losses[loss_key] = float(loss.mean().detach().cpu().numpy())
            if reset:
                del self.accumulated_train_losses
                self.accumulated_train_losses = defaultdict(list)

        elif mode == 'val':
            for loss_key, loss in self.accumulated_val_losses.items():
                loss = torch.stack(loss)
                averaged_losses[loss_key] = float(loss.mean().detach().cpu().numpy())
            if reset:
                del self.accumulated_val_losses
                self.accumulated_val_losses = defaultdict(list)

        return averaged_losses


