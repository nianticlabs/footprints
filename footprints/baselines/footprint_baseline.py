# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from skimage.morphology import convex_hull_image

from footprints.utils import readlines
from footprints.baselines.utils import BackprojectDepth, generate_camera_rays
from footprints.baselines.prepare_test_data import KittiTestLoader, MatterportTestLoader
from footprints.baselines.ransac import fit_plane, plane_distance


def safe_convex_hull_image(im):
    try:
        return convex_hull_image(im)
    except ValueError:
        print("Warning - failed to compute convex hull")
        return im * 0


class BaselineParentClass:

    def __init__(self, dataset_type):
        self.filenames = []
        self.dataset_type = dataset_type
        self.loader = {
            "kitti": KittiTestLoader,
            "matterport": MatterportTestLoader
            }[dataset_type](self.load_bounding_box_predictions,
                            self.load_visible_ground,
                            self.baseline_type)

    def run_all(self):
        """Apply baseline to all images in test set
        """
        for test_file_line in tqdm(self.filenames, desc=self.baseline_type):

            inputs = self.loader(test_file_line)

            ground_mask, ground_depth = self.frame_predict(inputs)

            save_path = self.loader.get_save_path(self.get_baseline_type(), test_file_line)

            cv2.imwrite(os.path.join(save_path + "_ground_mask.png"),
                        (ground_mask * 255).astype(np.uint8))

            if ground_depth is not None:
                np.save(os.path.join(save_path + "_ground_depth.npy"), ground_depth)

    def frame_predict(self, filename):
        raise NotImplementedError

    def get_baseline_type(self):
        return self.baseline_type

    def ransac_depth_inpaint(self, depth, inv_K, visible_ground_mask):

        backprojector = BackprojectDepth(*depth.shape)
        xyz = backprojector(depth, inv_K)

        m, _, _ = fit_plane(xyz[visible_ground_mask.ravel()])

        rays = generate_camera_rays(
            visible_ground_mask.shape[0],
            visible_ground_mask.shape[1],
            inv_K,
            ).T

        normalised_rays = rays / np.sqrt((rays**2).sum(1, keepdims=True))

        dot_product = np.sum(normalised_rays * m[:3][None, :], 1)
        distances_to_plane = plane_distance(m, xyz)  # distance of each point to the plane
        extra = distances_to_plane / dot_product
        plane_depth = depth - extra.reshape(depth.shape)
        return plane_depth


class RansacPlane(BaselineParentClass):
    load_bounding_box_predictions = False
    baseline_type = "ransac_plane"
    load_visible_ground = "pred"

    def frame_predict(self, inputs):
        visible_ground_mask = inputs["visible_ground"] > 0.5
        if visible_ground_mask.sum() < 20:
            return inputs['depth'], inputs['depth']
        floor_depth = self.ransac_depth_inpaint(
            inputs['depth'], inputs['inv_K'], visible_ground_mask)
        return floor_depth, floor_depth


class RansacPlaneOracle(RansacPlane):
    load_bounding_box_predictions = False
    baseline_type = "ransac_plane_oracle"
    load_visible_ground = "ground_truth"


class VisibleGround(BaselineParentClass):
    """Simplest baseline going: The hidden ground mask is the empty set.
    """
    baseline_type = "visible_ground"
    load_bounding_box_predictions = False
    load_visible_ground = "pred"

    def frame_predict(self, inputs):
        return inputs["visible_ground"] > 0.1, inputs.get("depth", None)


class ConvexHull(BaselineParentClass):
    """Second simplest baseline going:
        Total ground image is assumed to be the convex hull of the prediction.
        Depths are estimated as median per scanline
    """
    baseline_type = "convex_hull"
    load_bounding_box_predictions = False
    load_visible_ground = "pred"

    def frame_predict(self, inputs):

        # Compute the mask as the convex hull of the visible ground
        visible_ground_mask = inputs["visible_ground"] > 0.5
        all_predicted_floor_pixels = safe_convex_hull_image(visible_ground_mask)

        return all_predicted_floor_pixels, None


class BoundingBox(ConvexHull):
    """Use a 3D bounding box detection alg to find objects, and then project down to floor

    See baselines/README.md for how to reproduce the bounding box predictions on test images.
    """
    load_visible_ground = "pred"
    baseline_type = "bounding_box"

    def __init__(self, dataset_type, bounding_box_training_data):
        self.load_bounding_box_predictions = True
        super().__init__(dataset_type)
        self.bounding_box_training_data = bounding_box_training_data
        self.loader.bounding_box_training_data = bounding_box_training_data

    def frame_predict(self, inputs):

        visible_ground_mask = inputs["visible_ground"] > 0.5

        # Compute the mask as the convex hull of the visible ground
        all_floor_pixels = safe_convex_hull_image(visible_ground_mask)

        # remove the bounding box
        all_floor_pixels[inputs["bounding_box_mask"] < 0.5] = 0
        all_floor_pixels[visible_ground_mask == 1] = 1

        return all_floor_pixels, None

    def get_baseline_type(self):
        return self.baseline_type + "_" + self.bounding_box_training_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluation script for footprints.')

    parser.add_argument('--dataset', type=str,
                        help='dataset to evaluate on',
                        choices=["matterport", "kitti"],
                        required=True)
    parser.add_argument('--tiny',
                        action="store_true",
                        help='flag to only evaluate on a tiny subset; useful for debugging')

    opts = parser.parse_args()

    if opts.dataset == "matterport":
        test_path = "/mnt/nas/shared/datasets/matterport/splits/image_level/test.txt"
        test_filenames = readlines(test_path)
        test_filenames = test_filenames[:500]

    elif opts.dataset == "kitti":
        test_filenames = list(range(697))

    if opts.tiny:
        test_filenames = test_filenames[:20]

    print("Testing on {} images".format(len(test_filenames)))

    predictor = VisibleGround(opts.dataset)
    predictor.filenames = test_filenames
    predictor.run_all()

    predictor = ConvexHull(opts.dataset)
    predictor.filenames = test_filenames
    predictor.run_all()

    if opts.dataset == "matterport":

        # Note: Files required for matterport bounding box baseline are available on request

        # predictor = BoundingBox(opts.dataset, "sunrgbd")
        # predictor.filenames = test_filenames
        # predictor.run_all()

        # predictor = BoundingBox(opts.dataset, "scannet")
        # predictor.filenames = test_filenames
        # predictor.run_all()

        predictor = RansacPlaneOracle(opts.dataset)
        predictor.filenames = test_filenames
        predictor.run_all()

        predictor = RansacPlane(opts.dataset)
        predictor.filenames = test_filenames
        predictor.run_all()

    elif opts.dataset == "kitti":
        predictor = BoundingBox(opts.dataset, "3d_boundingbox")
        predictor.filenames = test_filenames
        predictor.run_all()
