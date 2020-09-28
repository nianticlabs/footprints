# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import argparse

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import time

cv2.setNumThreads(0)

from .data_loader import KITTILoader, MatterportLoader
from .geometry import *
from ...utils import *


class GroundTruthGenerator:
    """ Class to generate ground truth disparities of both visible and hidden ground"""

    def __init__(self, opts):

        self.opts = opts

        self.filenames = readlines(opts.textfile)
        self.filenames = sorted(self.filenames)
        if self.opts.idx_end == -1:
            self.filenames = self.filenames[self.opts.idx_start:]
        else:
            self.filenames = self.filenames[self.opts.idx_start:self.opts.idx_end]

        self.projector = BatchProjector(self.height, self.width)

        self.save_folder = opts.save_folder_name
        if self.save_folder is None:
            self.save_folder = 'hidden_depths'

        self.footprint_threshold = opts.footprint_threshold

    def load_data(self, idx, filename):
        """ Load and process data"""
        raise NotImplementedError

    def save_result(self, result, savepath, filename, save_viz=False):

        _savepath = os.path.join(savepath, 'data')
        os.makedirs(_savepath, exist_ok=True)
        np.save(os.path.join(_savepath, '{}.npy'.format(str(filename).zfill(10))), result)

        if self.opts.save_visualisations:
            _savepath = os.path.join(savepath, 'visualisations')
            os.makedirs(_savepath, exist_ok=True)
            plt.imsave(os.path.join(_savepath, '{}.jpg'.format(str(filename).zfill(10))), result)

    def parse_config(self, config, data_key):
        config = load_config(config)
        raw_data = config[data_key]['dataset']
        training_data = config[data_key]['training_data']
        return raw_data, training_data

    def compute_depth_mask(self, depth, ground_seg, K, invK):

        """Generate the depth mask for *untraversable* pixels"""

        ground_pix = ground_seg > self.footprint_threshold

        # project pixels to world
        world_points = self.projector.project_to_world(depth, invK).cpu().numpy()[0, :3].T

        # ransac fit plane to ground pix and find distance to plane
        ground_plane, _, _ = fit_plane(world_points[ground_pix.reshape(-1)])
        distances = plane_distance(ground_plane, world_points)

        # now move pixels distance along the normal vector
        normal = ground_plane[:3] / np.linalg.norm(ground_plane[:3])
        flattened_points = world_points - normal.reshape(1, 3) * distances.reshape(-1, 1)
        flattened_points = np.concatenate(
            [flattened_points, np.ones((flattened_points.shape[0], 1))], 1)

        # we only care about non ground pixels
        flattened_points = flattened_points[~ground_pix.reshape(-1)]

        # Splatting
        # find 2 orthogonal vectors which lie on the plane
        v1 = np.zeros(4)
        v2 = np.zeros(4)

        v1[:3] = np.cross(normal, np.array([0, 0, 1]))
        v2[:3] = np.cross(normal, v1[:3])

        v1 = v1.reshape(1, 4)
        v2 = v2.reshape(1, 4)

        # create a 2d grid of points around each flattened world point -> splatting
        points = []
        for d1 in np.arange(-0.1, 0.1, 0.025):
            for d2 in np.arange(-0.1, 0.1, 0.025):
                points.append(flattened_points + v1 * d1 + v2 * d2)

        points = torch.from_numpy(np.concatenate(points, 0).T).float().cuda()

        # project back into camera
        cam_pix = self.projector.project_to_camera(points, K, torch.eye(4).float().cuda())
        projection = self.projector.extract_depth_from_projections(cam_pix).cpu().numpy()[0]

        # filter -> must be pretty sure its not ground, within 10% of visible depth and closer
        # than 30 metres
        depth = depth.cpu().numpy()[0]
        result = (projection > 0) * (ground_seg < 0.5) * (
                (np.abs(projection - depth) / (depth + 1e-7)) < 0.10) * (projection < 30) * \
                 (depth > 0)

        return result

    def process_data(self, data, robust_aggregation=True):

        # project all frames into reference camera
        world_points = self.projector.project_to_world(data['depths'], data['inv_intrinsics'])
        cam_pix = self.projector.project_to_camera(world_points, data['poses'],
                                                   data['intrinsics'])
        projections = self.projector.extract_depth_from_projections(cam_pix).cpu().numpy()

        if robust_aggregation:
            # filter projections for agreement on pixels
            mask = (np.sum((projections > 0).astype(float), 0, keepdims=True) > 2)
            filtered_projections = projections * mask
        else:
            filtered_projections = projections

        masked = np.ma.MaskedArray(filtered_projections, mask=filtered_projections == 0)
        median = np.ma.median(masked, axis=0).filled(0)

        return median

    def run(self):

        time_before_batch = time.time()
        print('running ground truth generation on {} files...'.format(len(self.filenames)))
        for i, filename in enumerate(self.filenames):

            if i % 25 == 0:
                print('computing image {} of {}'.format(i, len(self.filenames)))
                if i != 0:
                    print(
                        'average time per image: {}'.format((time.time() - time_before_batch) / 25))
                    time_before_batch = time.time()
                    buffer = getattr(self.loader, 'buffer', None)
                    if buffer is not None:
                        print('buffer size {}'.format(len(self.loader.buffer)))

            data = self.load_data(i, filename)
            result = self.process_data(data, robust_aggregation=self.robust_aggregation)
            self.save_result(result, filename, save_viz=self.opts.save_visualisations)


class KITTIGroundTruthGenerator(GroundTruthGenerator):

    height, width = 192, 640

    def __init__(self, opts):

        super(KITTIGroundTruthGenerator, self).__init__(opts)

        self.raw_datapath, self.training_datapath = \
            self.parse_config(opts.config_path, data_key='kitti')

        self.sequence_in_buffer = None
        self.loader = KITTILoader(self.raw_datapath, self.training_datapath,
                                  self.height, self.width,
                                  footprint_threshold=self.footprint_threshold)
        self.robust_aggregation = True

    def load_data(self, idx, filename):

        sequence, frame, side = filename.split(' ')

        if sequence != self.sequence_in_buffer:
            self.loader.purge_buffer()
            self.sequence_in_buffer = sequence

        # ensure the buffer doesn't get too big
        if len(self.loader.buffer) > 1000:
            self.loader.purge_buffer()

        if side == 'l':
            side = 'image_02'
            stereo_baseline = self.loader.stereo_baseline
        else:
            side = 'image_03'
            stereo_baseline = self.loader.stereo_baseline * -1.0

        data = self.loader.load_data(sequence, int(frame))

        # only keep depths of ground pixels for projection
        data['depths'] = data['depths'] * data['ground_segs']

        # move to gpu
        data['depths'] = data['depths'].cuda()
        data['poses'] = data['poses'].cuda()
        data['intrinsics'] = data['intrinsics'].cuda()
        data['inv_intrinsics'] = data['inv_intrinsics'].cuda()

        # convert poses to relative pose
        base_pose = self.loader.load_frame_data(sequence, frame, side)['pose']
        base_pose = torch.from_numpy(np.linalg.pinv(base_pose)).float().unsqueeze(0).cuda()
        data['poses'] = torch.matmul(base_pose, data['poses'])

        # apply baseline to stereo cameras
        for frame_num in range(len(data['poses'])):
            if data['sides'][frame_num] != side:
                data['poses'][frame_num, 0, 3] += stereo_baseline

        return data

    def save_result(self, result, filename, save_viz=False):

        sequence, frame, side = filename.split(' ')

        if side == 'l':
            side = 'image_02'
        else:
            side = 'image_03'

        savepath = os.path.join(self.training_datapath, self.save_folder, sequence, side)
        super(KITTIGroundTruthGenerator, self).save_result(result, savepath, frame,
                                                           save_viz=save_viz)


class KITTIMovingObjectDetector(KITTIGroundTruthGenerator):

    def __init__(self, opts):

        super(KITTIMovingObjectDetector, self).__init__(opts)

        # overwrite standard savepath
        self.save_folder = opts.save_folder_name
        if self.save_folder is None:
            self.save_folder = 'moving_object_masks'

        # Unnecessary for moving object detection
        self.robust_aggregation = None

        # extract intrinsics from loader and convert to tensors for ease
        self.K = torch.from_numpy(self.loader.K).unsqueeze(0).float().cuda()
        self.invK = torch.from_numpy(self.loader.invK).unsqueeze(0).float().cuda()

    def load_data(self, idx, filename):

        sequence, frame, side = filename.split(' ')

        if sequence != self.sequence_in_buffer:
            self.loader.purge_buffer()
            self.sequence_in_buffer = sequence

        # ensure the buffer doesn't get too big
        if len(self.loader.buffer) > 1000:
            self.loader.purge_buffer()

        if side == 'l':
            side = 'image_02'
        else:
            side = 'image_03'

        base_data = self.loader.load_frame_data(sequence, int(frame), side, load_flow=True)

        # try to use the previous frame in time - if doesn't exist (base frame is the first frame
        # in a sequence) use forward in time
        lookup_data = self.loader.load_frame_data(sequence, int(frame)-1,
                                                  side, load_flow=True)
        if lookup_data is None:
            lookup_data = self.loader.load_frame_data(sequence, int(frame)+1, side,
                                                      load_flow=True)

        data = {'base_data': base_data,
                'lookup_data': lookup_data}
        return data

    def process_data(self, data, robust_aggregation=None):

        """ Override with logic to find moving objects rather than ground pixels"""

        base_data = data['base_data']
        lookup_data = data['lookup_data']

        T = np.matmul(np.linalg.pinv(lookup_data['pose']), base_data['pose'])

        # convert to tensors
        T = torch.from_numpy(T).float().unsqueeze(0).cuda()
        disp = torch.from_numpy(base_data['disparity']).float().unsqueeze(0).cuda()
        depth = float(self.K[0, 0, 0]) * self.loader.stereo_baseline / disp

        # project
        world_points = self.projector.project_to_world(depth, self.invK)
        cam_pix = self.projector.project_to_camera(world_points, T, self.K)

        # take pixel coords and reshape to find flow from depth
        cam_pix = cam_pix[0, :2].reshape(2, depth.shape[1], depth.shape[2]).cpu().numpy()
        x_pix, y_pix = np.meshgrid(np.arange(depth.shape[2]), np.arange(depth.shape[1]))
        cam_pix[0] -= x_pix
        cam_pix[1] -= y_pix

        # now compare flow to induced flow
        flow = base_data['flow']

        diff = cam_pix - flow
        norm = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1])
        moving_mask = norm > 3

        return moving_mask


class KITTIDepthMaskingGenerator(KITTIGroundTruthGenerator):

    """Generate mask of *untraversable* pixels using visible depth map and ground segmentation"""

    def __init__(self, opts):

        super(KITTIDepthMaskingGenerator, self).__init__(opts)

        # overwrite standard savepath
        self.save_folder = opts.save_folder_name
        if self.save_folder is None:
            self.save_folder = 'depth_masks'

        # Unnecessary for depth masking
        self.robust_aggregation = None

        # extract intrinsics from loader and convert to tensors for ease
        self.K = torch.from_numpy(self.loader.K).unsqueeze(0).float().cuda()
        self.invK = torch.from_numpy(self.loader.invK).unsqueeze(0).float().cuda()

    def load_data(self, idx, filename):

        sequence, frame, side = filename.split(' ')

        if side == 'l':
            side = 'image_02'
        else:
            side = 'image_03'

        data = self.loader.load_frame_data(sequence, int(frame), side, use_buffer=False,
                                           threshold_ground=False)

        return data

    def process_data(self, data, robust_aggregation=None):

        disparity = data['disparity']
        ground_seg = data['ground_seg']

        # check that we at least some ground pixels in the image
        if (ground_seg > self.footprint_threshold).sum() < 100:
            result = np.zeros([self.height, self.width])

        else:
            disp = torch.from_numpy(disparity).float().unsqueeze(0).cuda()
            depth = float(self.K[0, 0, 0]) * self.loader.stereo_baseline / disp
            result = self.compute_depth_mask(depth, ground_seg, self.K, self.invK)

        return result


class MatterportGroundTruthGenerator(GroundTruthGenerator):

    height, width = 480, 640

    def __init__(self, opts):

        super(MatterportGroundTruthGenerator, self).__init__(opts)

        self.raw_datapath, self.training_datapath = \
            self.parse_config(opts.config_path, data_key='matterport')

        self.loader = MatterportLoader(self.raw_datapath, self.training_datapath,
                                       self.height, self.width,
                                       footprint_threshold=self.footprint_threshold)
        self.robust_aggregation = False

    def load_data(self, idx, filename):

        scan, pos, height, direction = filename.split()

        data = self.loader.load_data(scan, pos, height, direction)

        # only keep depths of ground pixels for projections
        data['depths'] = data['depths'] * data['ground_segs']

        # move to gpu
        data['depths'] = data['depths'].cuda()
        data['poses'] = data['poses'].cuda()
        data['intrinsics'] = data['intrinsics'].cuda()
        data['inv_intrinsics'] = data['inv_intrinsics'].cuda()

        # filter for close cameras
        base_pose = self.loader.pose_tracker[(pos, height, direction)]
        inv_pose = torch.from_numpy(np.linalg.pinv(base_pose)).float().unsqueeze(0).cuda()
        base_pose = torch.from_numpy(base_pose).float().unsqueeze(0).cuda()
        filtered = (torch.abs(base_pose[:, 0, 3] - data['poses'][:, 0, 3]) < 10) * \
                   (torch.abs(base_pose[:, 1, 3] - data['poses'][:, 1, 3]) < 10) * \
                   (torch.abs(base_pose[:, 2, 3] - data['poses'][:, 2, 3]) < 1)

        data['poses'] = data['poses'][filtered]
        data['depths'] = data['depths'][filtered]
        data['intrinsics'] = data['intrinsics'][filtered]
        data['inv_intrinsics'] = data['inv_intrinsics'][filtered]

        # convert poses to relative pose
        data['poses'] = torch.matmul(inv_pose, data['poses'])

        return data

    def save_result(self, result, filename, save_viz=False):

        scan, pos, height, direction = filename.split()
        savepath = os.path.join(self.training_datapath, self.save_folder, scan)
        filename = '{}_{}_{}'.format(pos, height, direction)
        super(MatterportGroundTruthGenerator, self).save_result(result, savepath, filename,
                                                                save_viz=save_viz)


class MatterportDepthMaskingGenerator(MatterportGroundTruthGenerator):

    """Generate mask of *untraversable* pixels using visible depth map and ground segmentation"""

    def __init__(self, opts):

        super(MatterportDepthMaskingGenerator, self).__init__(opts)

        # overwrite standard savepath
        self.save_folder = opts.save_folder_name
        if self.save_folder is None:
            self.save_folder = 'depth_masks'

        # Unnecessary for moving object detection
        self.robust_aggregation = None

    def load_data(self, idx, filename):

        scan, pos, height, direction = filename.split()
        ground_seg, depth, _, K = self.loader.load_frame_data(scan, pos, height, direction)
        invK = np.linalg.pinv(K)

        data = {'depth': torch.from_numpy(depth).unsqueeze(0).float().cuda(),
                'ground_seg': ground_seg,
                'intrinsics': torch.from_numpy(K).unsqueeze(0).float().cuda(),
                'inv_intrinsics': torch.from_numpy(invK).unsqueeze(0).float().cuda()}
        return data

    def process_data(self, data, robust_aggregation=None):

        depth = data['depth']
        K = data['intrinsics']
        invK = data['inv_intrinsics']
        ground_seg = data['ground_seg']

        # check that we at least some ground pixels in the image
        if (ground_seg > self.footprint_threshold).sum() < 100:
            result = np.zeros([self.height, self.width])

        else:
            result = self.compute_depth_mask(depth, ground_seg, K, invK)

        return result


def get_options():
    """ parse command line options """
    parser = argparse.ArgumentParser(
        description='process frames to generate footprint training data')
    parser.add_argument('--config_path',
                        type=str,
                        help="path to config file containing dataset information",
                        default='paths.yaml')
    parser.add_argument('--type',
                        help='type of data to compute, either hidden depths (reprojected from other'
                             'views), moving object masks or depth masks (untraversable pixels)',
                        type=str,
                        choices=['hidden_depths', 'moving_objects', 'depth_masks'])
    parser.add_argument('--data_type',
                        type=str,
                        choices=['kitti', 'matterport'])
    parser.add_argument('--save_folder_name',
                        type=str,
                        help='folder name to save to - if not set, defaults to "hidden_depths"'
                             'if computing ground ground truth, and "moving_object_masks" if'
                             'computing moving object masks')
    parser.add_argument('--save_visualisations',
                        action='store_true',
                        help='if set, save images as well as npys')
    parser.add_argument('--textfile',
                        type=str,
                        help='textfile containing frames to be computed')
    parser.add_argument('--idx_start',
                        type=int,
                        help='allows for splitting generation into different threads/gpus by'
                             'starting/ending at specific indices',
                        default=0)
    parser.add_argument('--idx_end',
                        type=int,
                        help='allows for splitting generation into different threads/gpus by'
                             'starting/ending at specific indices',
                        default=-1)
    parser.add_argument('--footprint_threshold',
                        type=float,
                        default=0.75,
                        help='threshold for ground segmentation')

    return parser.parse_args()


if __name__ == '__main__':
    opts = get_options()
    if opts.data_type == 'kitti':
        if opts.type == 'hidden_depths':
            GTGenerator = KITTIGroundTruthGenerator(opts)
        elif opts.type == 'moving_objects':
            GTGenerator = KITTIMovingObjectDetector(opts)
        elif opts.type == 'depth_masks':
            GTGenerator = KITTIDepthMaskingGenerator(opts)
        else:
            raise NotImplementedError
    elif opts.data_type == 'matterport':
        if opts.type == 'hidden_depths':
            GTGenerator = MatterportGroundTruthGenerator(opts)
        elif opts.type == 'depth_masks':
            GTGenerator = MatterportDepthMaskingGenerator(opts)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    GTGenerator.run()
