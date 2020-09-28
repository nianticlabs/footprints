# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import matplotlib.pyplot as plt
from loguru import logger

from ..utils import normalise_image, depth_to_disp


def log(writer, inputs, outputs, losses, lr, step):
    writer.add_scalar('lr', lr, step)

    # write to tensorboard
    writer.add_scalar('loss', losses['loss'], step)

    for i in range(4):
        writer.add_image('image/{}'.format(i), inputs['image'][i], step)
        writer.add_image('target_disp/{}'.format(i),
                         normalise_image(depth_to_disp(inputs['depth'][i])),
                         step)
        writer.add_image('target_visible_ground/{}'.format(i),
                         normalise_image(inputs['visible_ground'][i]), step)
        writer.add_image('target_all_ground/{}'.format(i),
                         normalise_image(inputs['all_ground'][i]), step)
        writer.add_image('target_ground_depth/{}'.format(i),
                         normalise_image(inputs['ground_depth'][i]), step)
        writer.add_image('depth_mask/{}'.format(i),
                         normalise_image(inputs['depth_mask'][i]), step)

        moving_pix = inputs.get('moving_object_mask')
        if moving_pix is not None:
            writer.add_image('moving_pixels/{}'.format(i),
                             normalise_image(moving_pix[i]), step)

        for j in [1]:  # just log highest scale
            # colormap disparity
            disp = depth_to_disp(outputs[('depth', '1/{}'.format(j))][i])
            disp = normalise_image(disp).detach().cpu().numpy()
            disp_cm = plt.cm.plasma(disp)[..., :3].transpose([2, 0, 1])
            writer.add_image('pred_disp_{}/{}'.format(j, i), disp_cm, step)
            writer.add_image('pred_ground_visible_{}/{}'.format(j, i),
                             normalise_image(
                                 outputs[('visible_ground', '1/{}'.format(j))][i]),
                             step)

            writer.add_image('pred_ground_all_{}/{}'.format(j, i),
                             normalise_image(
                                 outputs[('all_ground', '1/{}'.format(j))][
                                     i]),
                             step)

            writer.add_image('pred_ground_disp_{}/{}'.format(j, i),
                             normalise_image(
                                 depth_to_disp(
                                     outputs[('ground_depth', '1/{}'.format(j))][
                                         i])),
                             step)

            writer.add_image('pred_ground_disp_masked_{}/{}'.format(j, i),
                             normalise_image(
                                 depth_to_disp(
                                     outputs[('ground_depth_masked', '1/{}'.format(j))][
                                     i])),
                             step)


class TimeLogger:

    def __init__(self):

        self.train_network_time = 0
        self.val_time = 0
        self.log_time = 0

    def _reset(self):
        self.train_network_time = 0
        self.val_time = 0
        self.log_time = 0

    def add_time(self, timer, time):
        _timer = self.__getattribute__(timer)
        _timer += time
        self.__setattr__(timer, _timer)

    def print_time(self):
        logger.info('{:.2f}s/{:.2f}s/{:.2f}s -- train/val/log'.format(
            self.train_network_time, self.val_time, self.log_time
        ))
        self._reset()
