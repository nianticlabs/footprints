# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from PIL import Image
import cv2

import torch

cv2.setNumThreads(0)


class BaseLoader:
    """ base class to fetch data for footprint ground truth generation """
    def __init__(self, raw_data_path, training_data_path, height, width, footprint_threshold=0.75):

        self.raw_data_path = raw_data_path
        self.training_data_path = training_data_path
        self.height = height
        self.width = width
        self.footprint_threshold = footprint_threshold

    def load_data(self):
        """ load depths, poses, ground segmentation, intrinsics and inverse intrinsics for
        frames to be projected """
        raise NotImplementedError

    def load_frame_data(self):
        """ load disparity, pose and footprint for a given frame """
        raise NotImplementedError


class KITTILoader(BaseLoader):

    """Class to load data for KITTI training data generation.

    For KITTI we need to define how maybe frames in the past and future to reproject from.
    Since each frame will be projected into many different frames, we load the data once, and
    store it in a buffer to be faster"""

    def __init__(self, raw_data_path, training_data_path, height, width, num_frames_bwd=25,
                 num_frames_fwd=50, footprint_threshold=0.75):

        super(KITTILoader, self).__init__(raw_data_path, training_data_path, height, width,
                                          footprint_threshold=footprint_threshold)

        self.num_frames_bwd = num_frames_bwd
        self.num_frames_fwd = num_frames_fwd

        self.buffer = {}  # buffer to keep depth, footprints, pose in memory
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.K[0] *= width
        self.K[1] *= height
        self.invK = np.linalg.pinv(self.K)
        self.stereo_baseline = 0.54

    def load_data(self, sequence, frame):
        """ load depths, poses, footprints for frames to be projected into 'frame' """
        disparities = []
        ground_segs = []
        poses = []
        sides = []
        intrinsics = []
        inv_intrinsics = []

        for frame_id in range(frame - self.num_frames_bwd, frame + self.num_frames_fwd, 2):
            for _side in ['image_02', 'image_03']:
                data = self.load_frame_data(sequence, frame_id, _side)

                if data:
                    disparities.append(data['disparity'])
                    ground_segs.append(data['ground_seg'])
                    poses.append(data['pose'])
                    sides.append(_side)
                    intrinsics.append(self.K)
                    inv_intrinsics.append(self.invK)

        disparities = torch.from_numpy(np.stack(disparities)).float()
        ground_segs = torch.from_numpy(np.stack(ground_segs)).float()
        poses = torch.from_numpy(np.stack(poses)).float()
        intrinsics = torch.from_numpy(np.stack(intrinsics)).float()
        inv_intrinsics = torch.from_numpy(np.stack(inv_intrinsics)).float()

        # convert disparities to depth and only keep ground pixels
        depths = self.K[0, 0] * self.stereo_baseline / disparities

        inputs = {'depths': depths,
                  'ground_segs': ground_segs,
                  'poses': poses,
                  'sides': sides,
                  'intrinsics': intrinsics,
                  'inv_intrinsics': inv_intrinsics}

        return inputs

    def load_frame_data(self, sequence, frame, side, load_flow=False, use_buffer=True,
                        threshold_ground=True):
        """ load disparity, pose and footprint for a given frame, either from disk or from the
        buffer. Additionally can load optical flow to compute moving object masks """

        # try to load from the buffer
        if use_buffer:
            data = self.buffer.get((sequence, frame, side))
        else:
            data = None

        if data:
            return data
        # otherwise load from disk and update the buffer
        else:
            try:
                # load disp, rescale and resize
                disp = np.load(os.path.join(self.training_data_path, 'stereo_matching_disps',
                                            sequence, side, '{}.npy'.format(str(frame).zfill(10))))
                disp *= (self.width / disp.shape[1])
                disp = cv2.resize(disp.astype(float), (self.width, self.height))

                # load segmentation and find ground
                ground_seg = np.load(os.path.join(self.training_data_path, 'ground_seg',
                                                  sequence, side, 'data',
                                                  '{}.npy'.format(str(frame).zfill(10))))[0]

                ground_seg = cv2.resize(ground_seg.astype(float), (self.width, self.height))
                if threshold_ground:
                    ground_seg = (ground_seg > self.footprint_threshold).astype(float)

                # load pose
                pose = np.eye(4)
                pose[:3] =\
                    np.load(os.path.join(self.training_data_path, 'poses', sequence,
                                         'orbslam_poses',
                                         '{}.npy'.format(str(frame).zfill(10)))).reshape(3, 4)

                # update buffer and return data
                data = {'disparity': disp,
                        'ground_seg': ground_seg,
                        'pose': pose,
                        }

                if load_flow:
                    flow = np.load(os.path.join(self.training_data_path, 'optical_flow', sequence,
                                                side, 'data',
                                                '{}.npy'.format(str(frame).zfill(10))))

                    # rescale and resize
                    resized_flow = np.zeros((2, self.height, self.width))
                    resized_flow[0] = cv2.resize(flow[0].astype(float),
                                                 (self.width, self.height)) * \
                                      self.width / flow.shape[2]
                    resized_flow[1] = cv2.resize(flow[1].astype(float),
                                                 (self.width, self.height)) * \
                                      self.height / flow.shape[1]

                    data.update({'flow': resized_flow})
                self.buffer[(sequence, frame, side)] = data

                return data

            except FileNotFoundError:
                # we are trying to load a nonexistent frame
                return None

    def purge_buffer(self):
        print('purging buffer!')
        del self.buffer
        self.buffer = {}


class MatterportLoader(BaseLoader):

    """ Class to load Matterport data for training data generation.

    For Matterport, we load all data from each house at once, and only project relevant frames.
    Data will be accessed from 'self.scan_data' unless it doesn't exist, in which case it will
    be loaded."""

    def __init__(self, raw_data_path, training_data_path, height, width, footprint_threshold=0.75):

        super(MatterportLoader, self).__init__(raw_data_path, training_data_path, height, width,
                                               footprint_threshold=footprint_threshold)

        self.current_scan = None
        self.scan_data = None
        self.pose_tracker = {}

        self.full_size_width = 1280.0
        self.full_size_height = 1024.0

    def load_data(self, scan, pos, height, direction):

        if self.current_scan != scan:
            # load all data for the new scan into memory
            self.pose_tracker = {}
            self.current_scan = scan
            self.load_scan_data()

        return self.scan_data.copy()

    def load_frame_data(self, scan, pos, height, direction, threshold_ground=True):

        scan_path = os.path.join(self.raw_data_path, scan, scan)
        ground_path = os.path.join(self.training_data_path, 'ground_seg',
                                          scan, 'data',
                                          '{}_{}_{}.npy'.format(pos, height, direction))
        ground_seg = (np.load(ground_path)[0] > self.footprint_threshold).astype(float)
        ground_seg = cv2.resize(ground_seg, (self.width, self.height),
                                interpolation=cv2.INTER_NEAREST)

        depth = Image.open(os.path.join(scan_path, 'matterport_depth_images',
                                        '{}_d{}_{}.png'.format(pos, height, direction)))

        depth = depth.resize((self.width, self.height), resample=Image.NEAREST)

        # rescale
        depth = np.array(depth).astype(float) * 0.00025

        with open(os.path.join(scan_path, 'matterport_camera_poses',
                               '{}_pose_{}_{}.txt'.format(pos, height, direction)), 'r') as fh:
            pose = np.array(fh.read().split()).astype(float).reshape(4, 4)

        intrinsics = np.eye(4)
        with open(os.path.join(scan_path, 'matterport_camera_intrinsics',
                               '{}_intrinsics_{}.txt'.format(pos, height)), 'r') as fh:
            intrinsics_from_file = fh.read().split()
            intrinsics[0, 0] = float(intrinsics_from_file[2])
            intrinsics[0, 2] = float(intrinsics_from_file[4])
            intrinsics[1, 1] = float(intrinsics_from_file[3])
            intrinsics[1, 2] = float(intrinsics_from_file[5])

            intrinsics[0] *= self.width / self.full_size_width
            intrinsics[1] *= self.height / self.full_size_height

        return ground_seg, depth, pose, intrinsics

    def load_scan_data(self):
        """ load depths, poses, footprints for all frames in a scan"""

        ground_segs = []
        depths = []
        poses = []
        intrinsics = []
        inv_intrinsics = []

        files = sorted(os.listdir(os.path.join(self.training_data_path, 'ground_seg',
                                               self.current_scan, 'data')))

        for idx, file in enumerate(files):
            if idx % 50 == 0:
                print('loaded {} of {}'.format(idx, len(files)))
            if file[-4:] == '.npy' and file[0] != '.':
                pos, height, direction = file.split('_')
                direction = direction[0]  # direction contains .npy extension

                ground_seg, depth, pose, K = self.load_frame_data(self.current_scan, pos,
                                                                  height, direction)

                ground_segs.append(ground_seg)
                depths.append(depth)
                poses.append(pose)
                intrinsics.append(K)
                inv_intrinsics.append(np.linalg.pinv(K))

                # store poses to change from absolute pose to relative pose
                self.pose_tracker[(pos, height, direction)] = pose

        depths = torch.from_numpy(np.stack(depths)).float()
        ground_segs = torch.from_numpy(np.stack(ground_segs)).float()
        poses = torch.from_numpy(np.stack(poses)).float()
        intrinsics = torch.from_numpy(np.stack(intrinsics)).float()
        inv_intrinsics = torch.from_numpy(np.stack(inv_intrinsics)).float()

        self.scan_data = {'depths': depths,
                          'ground_segs': ground_segs,
                          'poses': poses,
                          'intrinsics': intrinsics,
                          'inv_intrinsics': inv_intrinsics}

