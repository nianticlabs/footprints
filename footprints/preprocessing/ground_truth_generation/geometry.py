# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import numpy as np
import torch


class BatchProjector:

    """Class to batch forward warp depth values from multiple frames into a target frame"""

    def __init__(self, height, width):

        self.height = height
        self.width = width

        x_pix, y_pix = np.meshgrid(np.arange(width), np.arange(height))
        self.cam_points = np.stack([x_pix, y_pix, np.ones([height, width])])
        self.cam_points = torch.from_numpy(self.cam_points).unsqueeze(0).float().cuda()

    def project_to_world(self, depth, invK):

        batch_size = depth.shape[0]

        world_points = torch.matmul(invK[:, :3, :3], self.cam_points.reshape(1, 3, -1))
        world_points = world_points * depth.reshape(batch_size, 1, -1)
        world_points = torch.cat([world_points, torch.ones([batch_size, 1,
                                                            self.height * self.width]).cuda()], 1)

        # find only positive depth
        mask = (depth.reshape(batch_size, -1) > 0).float()
        world_points[:, 3] = mask

        return world_points

    def project_to_camera(self, world_points, T, K):

        # project to camera
        cam_pix = torch.matmul(K, torch.matmul(T, world_points))
        cam_pix[:, :2] /= (cam_pix[:, 2].unsqueeze(1) + 1e-7)

        return cam_pix

    def extract_depth_from_projections(self, cam_pix):

        # TODO: batch this
        projections = []
        for i in range(len(cam_pix)):
            projection = torch.zeros((self.height, self.width)).cuda()
            # ignore bad pixels
            _cam_pix = cam_pix[i]
            _cam_pix = _cam_pix[:, (_cam_pix[0] > 0) * (_cam_pix[0] < self.width) *
                                   (_cam_pix[1] > 0) * (_cam_pix[1] < self.height) *
                                   (_cam_pix[2] > 0) * (_cam_pix[3] > 0)]
            # grab depths
            projection[_cam_pix[1].long(), _cam_pix[0].long()] = _cam_pix[2].float()
            projections.append(projection)

        projections = torch.stack(projections)
        return projections


# RANSAC PLANE FITTING
"""
RANSAC code from https://github.com/falcondai/py-ransac

The MIT License (MIT)

Copyright (c) 2013 Falcon Dai

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz


def estimate(xyzs):
    axyz = augment(xyzs[:3])
    m = np.linalg.svd(axyz)[-1][-1, :]
    return m


def plane_distance(coeffs, xyz):
    return coeffs.dot(augment(xyz).T) / np.linalg.norm(coeffs[:3])


def is_inlier(coeffs, xyz, threshold):
    return np.abs(plane_distance(coeffs, xyz)) < threshold


def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations,
               stop_at_goal=False, random_seed=10):
    best_ic = 0
    best_model = None
    best_mask = None

    for i in range(max_iterations):

        idx = np.random.randint(data.shape[0], size=int(sample_size))
        s = data[idx, :]  # take the minmal sample set

        m = estimate(s)  # estimate model with this set

        # make inlier mask for this model
        inlier_mask = is_inlier(m, data)
        ic = inlier_mask.sum()

        if ic > best_ic:
            best_ic = ic
            best_model = m
            best_mask = inlier_mask
            if ic > goal_inliers and stop_at_goal:
                break

    # print('took iterations:', i + 1, 'best model:', best_model, 'explains:', best_ic)
    return best_model, best_ic, best_mask


def fit_plane(xyz):
    n = 100
    max_iterations = 100
    goal_inliers = n * 0.3
    threshold = 0.05

    # RANSAC
    m, b, inlier_mask = run_ransac(
        xyz, estimate, lambda x, y: is_inlier(x, y, threshold), 3, goal_inliers, max_iterations)

    return m, b, inlier_mask


def norm(x):
    return x / np.sqrt((x ** 2).sum())
