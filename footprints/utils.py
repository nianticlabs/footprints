# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import yaml
import hashlib
import zipfile
import urllib.request
from PIL import Image

import cv2

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

cv2.setNumThreads(0)


MODEL_DIR = "models"
GROUND_TRUTH_DIR = "ground_truth_files"


def pixel_disp_to_depth(disp, focal_length, baseline):
    """
    Convert pixel disparty to real world depth
    """
    depth = focal_length * baseline / (disp - (disp == 0))
    depth[depth < 0] = 0
    return depth


def sigmoid_to_depth(disp, min_depth=0.1, max_depth=100):
    """ Convert disparity to depth """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return depth


def depth_to_disp(depth):
    mask = (depth > 0).float()
    disp = 1 / (depth + 1e-7) * mask
    return disp


def readlines(filename):
    """ read lines of a text file """
    with open(filename, 'r') as file_handler:
        lines = file_handler.read().splitlines()
    return lines


def normalise_image(img):
    """ Normalize image to [0, 1] range for visualization """
    # img_max = float(img.max().cpu().data)
    # img_min = float(img.min().cpu().data)

    img_max = float(img.max())
    img_min = float(img.min())
    denom = img_max - img_min if img_max != img_min else 1e5
    return (img - img_min) / denom


def sec_to_hm(secs):
    """ convert seconds to hours, minutes and seconds """
    secs = int(secs)
    seconds = secs % 60
    secs //= 60
    minutes = secs % 60
    hours = secs // 60
    return hours, minutes, seconds


def sec_to_hm_str(secs):
    """ convert seconds to a string of hours, minutes and seconds """
    hours, minutes, seconds = sec_to_hm(secs)
    return "{:02d}h{:02d}m{:02d}s".format(hours, minutes, seconds)


def load_config(config):
    with open(config, 'r') as fh:
        config = yaml.safe_load(fh)
    return config


def pil_loader(path):
    with open(path, 'rb') as file_handler:
        with Image.open(file_handler) as img:
            return img.convert('RGB')


def check_file_matches_md5(checksum, fpath):
    if not os.path.exists(fpath):
        return False
    with open(fpath, 'rb') as f:
        current_md5checksum = hashlib.md5(f.read()).hexdigest()
    return current_md5checksum == checksum


def download_model_if_doesnt_exist(model_name):
    """If pretrained model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "kitti": (
            "https://storage.googleapis.com/niantic-lon-static/research/footprints/kitti.zip",
            "a52e3b04bffd86f62c62cf8859c47798"),
        "matterport": (
            "https://storage.googleapis.com/niantic-lon-static/research/footprints/matterport.zip",
            "e28929d0819392d2178c880725531c4e"),
        "handheld": (
            "https://storage.googleapis.com/niantic-lon-static/research/footprints/handheld.zip",
            "ab97945cf8f8f9e8d9bdedf8961506b6"),
        }

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, model_name)

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "model.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("Failed to download a file which matches the checksum - quitting")
            quit()

        print("Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("Model unzipped to {}".format(model_path))


def download_ground_truths_if_dont_exist(dataset_name):
    """If ground truth files doen't exist, download and unzip them
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "kitti": (
            "https://storage.googleapis.com/niantic-lon-static/research/footprints/data/kitti/kitti_ground_truth.zip",
            "1e25ee18016a9a4a939219fcc56f6eba"),
        "matterport": (
            "https://storage.googleapis.com/niantic-lon-static/research/footprints/data/matterport/matterport_ground_truth.zip",
            "eb9e0f8a04e35ddd8aa3eda9079c6b17"),
        }

    os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)
    ground_truth_path = os.path.join(GROUND_TRUTH_DIR, "{}_ground_truth".format(dataset_name))
    os.makedirs(ground_truth_path, exist_ok=True)

    # see if we have files already extracted
    ground_truth_subdir = os.path.join(ground_truth_path, "{}_ground_truth".format(dataset_name))
    if not (os.path.exists(ground_truth_subdir) and len(os.listdir(ground_truth_subdir)) > 500):

        ground_truths_url, required_md5checksum = download_paths[dataset_name]

        if not check_file_matches_md5(required_md5checksum, ground_truth_path + ".zip"):
            print("Downloading ground truths to {}".format(ground_truth_path + ".zip"))
            urllib.request.urlretrieve(ground_truths_url, ground_truth_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, ground_truth_path + ".zip"):
            print("Failed to download a file which matches the checksum - quitting")
            quit()

        print("Unzipping ground truth files...")
        with zipfile.ZipFile(ground_truth_path + ".zip", 'r') as f:
            f.extractall(ground_truth_path)

        print("Ground truths for {} unzipped to {}".format(dataset_name, ground_truth_path))
