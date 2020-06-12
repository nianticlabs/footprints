# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import hashlib
import urllib
import zipfile
from PIL import Image

MODEL_DIR = "models"


def sigmoid_to_depth(disp, min_depth=0.1, max_depth=100):
    """ Convert disparity to depth """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return depth


def pil_loader(path):
    with open(path, 'rb') as file_handler:
        with Image.open(file_handler) as img:
            return img.convert('RGB')


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

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

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
