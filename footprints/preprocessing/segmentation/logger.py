# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import numpy as np
import matplotlib.pyplot as plt


COLORMAP = plt.get_cmap('plasma', 256)  # for plotting


def normalize_image(img):
    """ normalize image to [0,1] range for visualization """
    img_max = float(img.max())
    img_min = float(img.min())
    denom = img_max - img_min if img_max != img_min else 1e5
    return (img - img_min) / denom


def log(writer, inputs, prediction, losses, lr, step, num_outputs=10):
    # Write losses and lr
    for loss_type, loss_val in losses.items():
        writer.add_scalar("{}".format(loss_type), loss_val, step)
    writer.add_scalar('learning rate', lr, step)

    for i in range(min(num_outputs, inputs['image'].shape[0])):
        # Save images
        image = inputs['image'][i].float().detach().cpu().numpy()
        gt_ground = inputs['ground_mask'][i].float().cpu().numpy()
        pred_ground = prediction['ground'][i].float().detach().cpu().numpy()

        image = normalize_image(image)
        gt_colormapped = np.zeros_like(image)
        gt_colormapped[0, gt_ground == 1] = 1

        pred_ground = np.transpose(COLORMAP(pred_ground)[:, :, :3], [2, 0, 1])

        result = np.concatenate((image, gt_colormapped, pred_ground), axis=2)

        writer.add_image("Image/{}".format(i), result, step)
