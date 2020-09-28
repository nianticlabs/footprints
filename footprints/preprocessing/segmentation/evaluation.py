# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from collections import defaultdict

import torch
from torch import nn


class Evaluator:
    """
    Class to compute and track losses and evaluation metrics.

    tracked_loss will store loss metrics for different batches, which is reset each time we access
    it - used to accumulate losses/metrics over multiple batches
    """
    def __init__(self):
        self.loss_func = nn.BCEWithLogitsLoss(reduction='none')
        self.tracked_loss = defaultdict(list)

    def _reset_losses(self):
        self.tracked_loss = defaultdict(list)

    def get_tracked_losses(self):
        """ Get the accumulated losses and reset the tracking """
        losses = self.tracked_loss.copy()
        self._reset_losses()

        for key, val in losses.items():
            losses[key] = torch.cat(val, dim=0).mean()

        return losses

    def compute_losses(self, predictions, ground_mask, loss_mask):
        """ Compute loss and evaluation metrics for a batch and store in tracked_loss - also returns
        the batch loss to be used for backprop.

        Predicitons will be a dict of predictions for each scale """
        total_loss = 0

        for prediction_type, scale in predictions.keys():

            pred = predictions[(prediction_type, scale)]
            target = ground_mask
            # compute losses at each scale, mean over all except batch
            loss = self.loss_func(pred, target)
            valid_pix = loss_mask.sum(dim=[1, 2])
            loss = (loss * loss_mask).sum(dim=[1, 2]) / (valid_pix + 1e-7)

            self.tracked_loss['{}_loss_{}'.format(prediction_type, scale)].append(loss)
            total_loss += loss

        total_loss /= 4
        self.tracked_loss['loss'].append(total_loss)

        return total_loss.mean()
