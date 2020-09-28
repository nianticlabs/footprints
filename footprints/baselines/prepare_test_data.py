# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import cv2
import yaml
import numpy as np

from footprints.utils import sigmoid_to_depth
from footprints.utils import download_ground_truths_if_dont_exist, GROUND_TRUTH_DIR

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paths.yaml")) as f:
    paths = yaml.safe_load(f)


def cv2_imread_strict(im_path, *args):
    if os.path.isfile(im_path):
        return cv2.imread(im_path, *args)[:, :, ::-1]
    else:
        raise FileNotFoundError(im_path)


class TestLoader:
    """Class to load bits for baseline computation
    """
    def __init__(self,
                 load_bounding_box_predictions=False,
                 load_visible_ground=False,
                 baseline_type=""):
        self.load_bounding_box_predictions = load_bounding_box_predictions
        self.load_visible_ground = load_visible_ground
        self.baseline_type = baseline_type


class KittiTestLoader(TestLoader):
    """KITTI
    """
    pred_visible_ground_base_path = os.path.join(
        paths["kitti"]["predictions"], "ours", "{:03d}_color.npy")

    download_ground_truths_if_dont_exist("kitti")
    ground_truth_dir = os.path.join(GROUND_TRUTH_DIR, "kitti_ground_truth", "kitti_ground_truth")
    visible_ground_base_path = os.path.join(ground_truth_dir, "{:05d}_ground.png")

    W = 640
    H = 192

    def __call__(self, frame_num):

        inputs = {}

        if self.load_visible_ground == "pred":
            inputs["visible_ground"] = np.load(
                self.pred_visible_ground_base_path.format(frame_num))[0]  # VISIBLE_GROUND is 0th
        elif self.load_visible_ground == "ground_truth":
            inputs["visible_ground"] = cv2_imread_strict(
                self.visible_ground_base_path.format(frame_num))

        if self.load_bounding_box_predictions:
            bounding_box_path = os.path.join(
                paths["kitti"]["predictions"],
                "bounding_box_detections",
                "{:03d}_colorfootprint.png".format(frame_num))
            inputs["bounding_box_mask"] = cv2_imread_strict(bounding_box_path)[:, :, 0]

        for key in inputs:
            inputs[key] = cv2.resize(inputs[key], (self.W, self.H))

        return inputs

    def get_save_path(self, baseline_type, test_file_line):
        save_path = os.path.join(paths["kitti"]["predictions"],
                                 "..",
                                 "predictions_rerun",
                                 baseline_type,
                                 str(test_file_line))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path


class MatterportTestLoader(TestLoader):
    """Loading the matterport test set
    """
    ground_truth_dir = os.path.join(
        GROUND_TRUTH_DIR, "matterport_ground_truth", "matterport_ground_truth")
    visible_ground_base_path = os.path.join(
        ground_truth_dir, "{}_{}_{}_{}_groundtruth.npy")

    intrinsics_base_path = os.path.join(
        paths['matterport']['dataset'], "{}/{}/matterport_camera_intrinsics/{}_intrinsics_{}.txt")
    our_preds_base_path = os.path.join(
        paths["matterport"]["predictions"], "lambda_0.5", "{}_{}_{}_{}.npy")

    H = 512
    W = 640

    def load_intrinsics(self, frame_data, depth):

        int_path = self.intrinsics_base_path.format(frame_data[0], *frame_data)
        intrinsics = np.loadtxt(int_path)
        K = np.eye(3)
        K[0, 0] = float(intrinsics[2])
        K[1, 1] = float(intrinsics[3])
        K[0, 2] = float(intrinsics[4])
        K[1, 2] = float(intrinsics[5])

        K[0, :] /= self.W
        K[0, :] *= depth.shape[1]
        K[1, :] /= self.H
        K[1, :] *= depth.shape[0]
        inv_K = np.linalg.pinv(K)

        return K, inv_K

    def __call__(self, test_file_line):

        frame_data = test_file_line.strip().split()

        if 'ransac_plane' in self.baseline_type:
            pred = np.load(self.our_preds_base_path.format(*frame_data))
            depth = cv2.resize(sigmoid_to_depth(pred[2]), (self.W, self.H))

            K, inv_K = self.load_intrinsics(frame_data, depth)

            inputs = {"depth": depth, "inv_K": inv_K, "K": K}
        else:
            inputs = {}

        if self.load_visible_ground == "pred":
            pred = np.load(self.our_preds_base_path.format(*frame_data))
            inputs["visible_ground"] = cv2.resize(pred[0], (self.W, self.H))
        elif self.load_visible_ground == "ground_truth":
            gt = np.load(self.visible_ground_base_path.format(*frame_data))
            inputs["visible_ground"] = cv2.resize(gt, (self.W, self.H))

        if self.load_bounding_box_predictions:
            bbox_mask_path = self.bounding_box_mask_base_path.format(
                self.bounding_box_training_data, *frame_data)
            inputs["bounding_box_mask"] = cv2_imread_strict(bbox_mask_path)
            inputs["bounding_box_mask"] = cv2.resize(
                inputs["bounding_box_mask"], (self.W, self.H))[:, :, 0]

        return inputs

    def get_save_path(self, baseline_type, test_file_line):
        save_path = os.path.join(paths["matterport"]["predictions"],
                                 "..",
                                 "predictions_rerun",
                                 baseline_type,
                                 str(test_file_line).replace(' ', '_'))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path
