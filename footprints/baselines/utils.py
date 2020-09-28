# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import numpy as np


def norm(x):
    return x / np.sqrt((x**2).sum())


def generate_camera_rays(h, w, inv_K):
    """Creates a camera ray image, giving a 3D ray for each pixel
    """
    id_coords = np.stack(
        np.meshgrid(np.arange(w), np.arange(h), indexing='xy'), axis=0)
    ones = np.ones((1, h * w))
    pix_coords = np.stack([id_coords[0].ravel(), id_coords[1].ravel()])
    pix_coords = np.concatenate([pix_coords, ones], 0)
    cam_rays = inv_K[:3, :3].dot(pix_coords)

    return cam_rays


class BackprojectDepth:
    """Class to transform a single depth image into a point cloud
    """
    def __init__(self, height, width):

        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.ones = np.ones((1, self.height * self.width))

        self.pix_coords = np.stack([self.id_coords[0].ravel(), self.id_coords[1].ravel()])
        self.pix_coords = np.concatenate([self.pix_coords, self.ones], 0)

    def __call__(self, depth, inv_K):
        cam_points = inv_K[:3, :3].dot(self.pix_coords)
        cam_points = depth.reshape(1, -1) * cam_points
        return cam_points.T


class Project3D:
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.height = height
        self.width = width
        self.eps = eps

    def __call__(self, points, K, T):
        P = K.dot(T)[:3, :]

        cam_points = P.dot(points)

        pix_coords = cam_points[:2, :] / (cam_points[2, None, :] + self.eps)
        return pix_coords
