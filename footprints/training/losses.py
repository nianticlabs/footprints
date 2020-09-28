# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


import torch
from torch.nn import BCEWithLogitsLoss, Sigmoid

from ..utils import sigmoid_to_depth


class LossManager:
    """ Class to compute losses for AllInOne data - i.e. data with ground truth labels for
    visible ground, visible depth, hidden ground and hidden depth"""

    def __init__(self, depth_range, footprint_prior_weight):

        self.min_depth, self.max_depth = depth_range
        self.footprint_prior_weight = footprint_prior_weight

        self.depth_loss = SupervisedDepthLoss()
        self.bce_loss = BinaryCrossEntropy()

        self.hidden_footprint_loss = \
            ThreeClassLoss(prior_weight=self.footprint_prior_weight)

        self.sigmoid = Sigmoid()

    def __call__(self, predictions, targets):

        all_scale_loss = 0
        losses = {}
        outputs = {}

        target_depth = targets['depth']
        valid_depth = (target_depth > 0).float()

        target_ground = targets['visible_ground']
        target_ground_all = targets['all_ground']

        # mask is 1 if moving, so invert
        moving_object_mask = 1 - (targets['moving_object_mask'])

        target_ground_depth = targets['ground_depth']
        valid_depth_ground = (target_ground_depth > 0).float()

        depth_mask = targets['depth_mask']

        for scale_key, output in predictions.items():

            # Visible ground pixels - first channel
            pred = output[:, 0]
            losses[('visible_ground', scale_key)] = self.bce_loss(pred, target_ground)
            outputs[('visible_ground', scale_key)] = self.sigmoid(pred)  # apply sigmoid for viz

            # All ground pixels - second channel
            pred = output[:, 1]
            losses[('all_ground', scale_key)] = self.hidden_footprint_loss(pred, target_ground_all,
                                                                           depth_mask,
                                                                           moving_object_mask)
            outputs[('all_ground', scale_key)] = self.sigmoid(pred)  # apply sigmoid for viz

            # Depth - third channel
            pred = output[:, 2]
            pred = sigmoid_to_depth(pred, min_depth=self.min_depth, max_depth=self.max_depth)
            losses[('depth', scale_key)] = self.depth_loss(pred, target_depth, mask=valid_depth)
            outputs[('depth', scale_key)] = pred

            # All ground depth - fourth channel
            pred = output[:, 3]
            pred = sigmoid_to_depth(pred, min_depth=self.min_depth, max_depth=self.max_depth)
            losses[('ground_depth', scale_key)] = self.depth_loss(pred, target_ground_depth,
                                                                  mask=valid_depth_ground)
            outputs[('ground_depth', scale_key)] = pred
            outputs[('ground_depth_masked', scale_key)] = \
                pred * (outputs[('all_ground', scale_key)] > 0.5).float()

            losses[('loss', scale_key)] = losses[('depth', scale_key)] + \
                                          losses[('visible_ground', scale_key)] + \
                                          losses[('all_ground', scale_key)] + \
                                          losses[('ground_depth', scale_key)]

            all_scale_loss += losses[('loss', scale_key)]

        all_scale_loss /= 4  # mean over scales
        losses['loss'] = all_scale_loss

        predictions.update(outputs)

        return losses


class SupervisedDepthLoss:

    def __init__(self):
        pass

    def __call__(self, pred_depth, target_depth, mask=None):

        if mask is None:
            mask = torch.ones_like(target_depth)

        loss = torch.log(torch.abs(pred_depth - target_depth) + 1) * mask

        return loss.mean()


class BinaryCrossEntropy:

    def __init__(self):

        self.loss_func = BCEWithLogitsLoss(reduction='none')

    def __call__(self, pred, target, mask=None, take_mean=True):

        if mask is None:
            mask = torch.ones_like(pred)

        loss = self.loss_func(pred, target) * mask

        if take_mean:
            loss = loss.mean()

        return loss


class ThreeClassLoss:

    def __init__(self, prior_weight):
        self.prior_weight = prior_weight
        self.loss_func = BinaryCrossEntropy()

    def __call__(self, pred, ground_target, depth_mask, moving_mask):

        loss = 0

        # ground vs def not ground
        mask = ((ground_target + depth_mask) > 0).float()
        loss += self.loss_func(pred, ground_target, mask=mask, take_mean=False)

        # mask out moving objects, keeping loss on definitely not ground (happens in dataloader)
        if moving_mask is not None:
            loss *= moving_mask

        # now prior
        mask = (1 - mask)
        loss += self.prior_weight * self.loss_func(pred, torch.zeros_like(pred), mask=mask,
                                                   take_mean=False)

        return loss.mean()
