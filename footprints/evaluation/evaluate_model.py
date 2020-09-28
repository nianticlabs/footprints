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

from ..utils import download_ground_truths_if_dont_exist, GROUND_TRUTH_DIR


# Channels in the .npy prediction arrays
VISIBLE_GROUND = 0
HIDDEN_GROUND = 1
DEPTH = 2
HIDDEN_DEPTH = 3


def sigmoid_to_depth(disp, min_depth=0.1, max_depth=100):
    """Convert disparity to depth
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return depth


def readlines(filename):
    """Read lines of a text file and return as a list
    """
    with open(filename, 'r') as file_handler:
        return file_handler.read().splitlines()


def load_mask(filepath):
    """Load an image file as a binary mask
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) > 128


def evaluate_depth(gt, pred, max_depth=20):
    """Evaluate a single predicted depth image
    """
    gt = np.clip(gt, 0.5, max_depth)
    pred = np.clip(pred, 0.5, max_depth)

    if gt.size == 0:
        # no hidden ground pixels so we can ignore this
        return {key: np.nan for key in ["a1", "abs_rel", "sq_rel", "rmse"]}

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return {"a1": a1, "abs_rel": abs_rel, "sq_rel": sq_rel, "rmse": rmse}


def evaluate_mask(true, pred):
    """Evaluate a single predicted mask image compared with the ground truth mask
    """
    # forcing to be binary
    true_mask = true > 0.1  # ground truth marks anything non-zero as being hidden floor
    pred_mask = pred > 0.5  # prediction is softmax output, so threshold at 0.5

    if true_mask.sum() == 0:
        return {key: np.nan for key in ["iou", "precision", "recall", "f1"]}

    union = np.logical_or(true_mask, pred_mask)
    intersection = np.logical_and(true_mask, pred_mask)

    intersection_sum = intersection.sum()
    union_sum = union.sum()
    iou = intersection_sum / union_sum if union_sum > 0 else 0

    # true positive, false positive, false negative
    tp = np.logical_and(true_mask, pred_mask).sum()
    fp = np.logical_and(~true_mask, pred_mask).sum()
    fn = np.logical_and(true_mask, ~pred_mask).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}


def load_kitti_ground_truth(im_idx):
    """Load a single ground truth mask (and free space evaluation area) for the kitti dataset
    """
    ground_truth_dir = os.path.join(GROUND_TRUTH_DIR, "kitti_ground_truth", "kitti_ground_truth")
    ground_truth = load_mask(os.path.join(ground_truth_dir, "{:05d}_combined.png".format(im_idx)))
    free_space = load_mask(os.path.join(ground_truth_dir, "{:05d}_ground.png".format(im_idx)))
    return ground_truth, free_space


def safe_convex_hull_image(im):
    try:
        return convex_hull_image(im)
    except ValueError:
        print("Warning - failed to compute convex hull")
        return im * 0


def load_matterport_ground_truth(filename):
    """Load a single ground truth mask (and free space evaluation area) for the matterport dataset
    """
    ground_truth_dir = os.path.join(
        GROUND_TRUTH_DIR, "matterport_ground_truth", "matterport_ground_truth")

    hidden_ground = np.load(os.path.join(
        ground_truth_dir, "{}_{}_{}_{}_groundtruth.npy".format(*filename)))
    free_space = np.load(os.path.join(
        ground_truth_dir, "{}_{}_{}_{}_freespace.npy".format(*filename))) > 0.5
    return hidden_ground, free_space


def evaluate(pred_folder, datatype, metric):
    """Evaluate a folder of predictions for either matterport or kitti datasets
    """
    if datatype == "kitti":
        download_ground_truths_if_dont_exist("kitti")
        filenames = range(697)  # kitti files are simply indexed by an integer

        if metric == "depth":
            raise ValueError("The kitti annotations do not contain depth data for evaluation")

    elif datatype == "matterport":
        download_ground_truths_if_dont_exist("matterport")
        filenames = [xx.split() for xx in readlines(
            os.path.join("splits/matterport/test.txt"))]

    all_scores = []

    for filename in tqdm(filenames):

        if datatype == "kitti":
            ground_truth, free_space = load_kitti_ground_truth(filename)
            try:
                pred = np.load(
                    os.path.join(pred_folder, "{:03d}.npy".format(filename)))
            except FileNotFoundError:
                pred = load_mask(
                    os.path.join(pred_folder, "{:d}_ground_mask.png".format(filename)))

        elif datatype == "matterport":
            ground_truth, free_space = load_matterport_ground_truth(filename)
            pred = np.load(os.path.join(pred_folder, "{}".format(filename[0]),
                                        "{}_{}_{}.npy".format(*filename[1:])))

        if metric == "iou":
            if pred.ndim == 3:
                pred = pred[HIDDEN_GROUND]  # extract just the hidden ground channel

            all_scores.append({
                "freespace": evaluate_mask(ground_truth, pred),
                "footprint": evaluate_mask(1 - ground_truth[free_space], 1 - pred[free_space]),
                })

        elif metric == "depth":
            if pred.ndim == 3:
                pred = sigmoid_to_depth(pred[HIDDEN_DEPTH])  # extract just the depth channel

            mask = ground_truth > 0
            all_scores.append(evaluate_depth(ground_truth[mask], pred[mask]))

        else:
            raise Exception("unknown metric {}".format(metric))

    if metric == "iou":
        print("Freespace IoU:  {:0.3f}".format(
            np.nanmean([score['freespace']['iou'] for score in all_scores])))
        print("Freespace F1:   {:0.3f}".format(
            np.nanmean([score['freespace']['f1'] for score in all_scores])))
        print("Footprint IoU:  {:0.3f}".format(
            np.nanmean([score['footprint']['iou'] for score in all_scores])))
        print("Footprint F1:   {:0.3f}".format(
            np.nanmean([score['footprint']['f1'] for score in all_scores])))

    elif metric == "depth":
        print("a1:       {:0.3f}".format(np.nanmean([score['a1'] for score in all_scores])))
        print("rmse:     {:0.3f}".format(np.nanmean([score['rmse'] for score in all_scores])))
        print("Abs. rel: {:0.3f}".format(np.nanmean([score['abs_rel'] for score in all_scores])))
        print("Sq. rel:  {:0.3f}".format(np.nanmean([score['sq_rel'] for score in all_scores])))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple prediction from a footprints model.')

    parser.add_argument('--predictions', type=str,
                        help='path to folder of predictions', required=True)
    parser.add_argument('--datatype', type=str,
                        help='name of datatype to use',
                        choices=["kitti", "matterport"],
                        required=True)
    parser.add_argument('--metric', type=str,
                        help='what channel to evaluate',
                        choices=["iou", "depth"],
                        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(pred_folder=args.predictions, datatype=args.datatype, metric=args.metric)
