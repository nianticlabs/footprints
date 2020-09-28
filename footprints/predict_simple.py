# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import cv2
import glob
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from footprints.model_manager import ModelManager
from footprints.utils import sigmoid_to_depth, download_model_if_doesnt_exist, pil_loader, MODEL_DIR


MODEL_HEIGHT_WIDTH = {
    "kitti": (192, 640),
    "matterport": (512, 640),
    "handheld": (256, 448),
}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


class InferenceManager:

    def __init__(self, model_name, save_dir, use_cuda, save_visualisations=True):

        download_model_if_doesnt_exist(model_name)
        model_load_folder = os.path.join(MODEL_DIR, model_name)
        self.model_manager = ModelManager(is_inference=True, use_cuda=use_cuda)
        self.model_manager.load_model(weights_path=model_load_folder)
        self.model_manager.model.eval()

        self.use_cuda = use_cuda
        self.colormap = plt.get_cmap('plasma', 256)  # for plotting
        self.resizer = transforms.Resize(MODEL_HEIGHT_WIDTH[model_name],
                                         interpolation=Image.ANTIALIAS)
        self.totensor = transforms.ToTensor()

        self.save_dir = save_dir
        os.makedirs(os.path.join(save_dir,  "outputs"), exist_ok=True)
        self.save_visualisations = save_visualisations
        if save_visualisations:
            os.makedirs(os.path.join(save_dir,  "visualisations"), exist_ok=True)

    def _load_and_preprocess_image(self, image_path):
        """Load an image, resize it, convert to torch and if needed put on GPU
        """
        original_image = pil_loader(image_path)
        preprocessed_image = self.resizer(original_image)
        preprocessed_image = self.totensor(preprocessed_image)
        preprocessed_image = preprocessed_image[None, ...]
        if self.use_cuda:
            preprocessed_image = preprocessed_image.cuda()
        return original_image, preprocessed_image

    def predict_for_single_image(self, image_path):
        """Use the model to predict for a single image and save results to disk
        """
        print("Predicting for {}".format(image_path))
        original_image, preprocessed_image = self._load_and_preprocess_image(image_path)
        pred = self.model_manager.model(preprocessed_image)
        pred = pred['1/1'].data.cpu().numpy().squeeze(0)

        filename, _ = os.path.splitext(os.path.basename(image_path))
        npy_save_path = os.path.join(self.save_dir, "outputs", filename + '.npy')
        print("└> Saving predictions to {}".format(npy_save_path))
        np.save(npy_save_path, pred)

        if self.save_visualisations:

            hidden_ground = cv2.resize(pred[1], original_image.size) > 0.5
            hidden_depth = cv2.resize(sigmoid_to_depth(pred[3]), original_image.size)
            original_image = np.array(original_image) / 255.0

            # normalise the relevant parts of the depth map and apply colormap
            _max = hidden_depth[hidden_ground].max()
            _min = hidden_depth[hidden_ground].min()
            hidden_depth = (hidden_depth - _min) / (_max - _min)
            depth_colourmap = self.colormap(hidden_depth)[:, :, :3]  # ignore alpha channel

            # create and save visualisation image
            hidden_ground = hidden_ground[:, :, None]
            visualisation = original_image * (1 - hidden_ground) + depth_colourmap * hidden_ground
            vis_save_path = os.path.join(self.save_dir, "visualisations", filename + '.jpg')
            print("└> Saving visualisation to {}".format(vis_save_path))
            cv2.imwrite(vis_save_path, (visualisation[:, :, ::-1] * 255).astype(np.uint8))

    def predict_for_folder(self, folder_path):
        """Search through a folder of images for image files and predict for each one
        """
        paths = glob.glob(os.path.join(folder_path, '*'))
        for path in paths:
            if os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS:
                self.predict_for_single_image(path)

    def predict(self, image_path):
        """Determine if the prediction path is a folder or a single image, and handle appropriately
        """
        if os.path.isfile(image_path):
            self.predict_for_single_image(image_path)
        elif os.path.isdir(image_path):
            self.predict_for_folder(image_path)
        else:
            raise Exception("Can not find args.image: {}".format(image_path))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple prediction from a footprints model.')

    parser.add_argument('--image', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model', type=str,
                        help='name of a pretrained model to use',
                        choices=["kitti", "matterport", "handheld"])
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--no_save_vis",
                        help='if set, disables visualisation saveing',
                        action='store_true')
    parser.add_argument("--save_dir", type=str,
                        help='where to save npy and visualisations to',
                        default="predictions")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    inference_manager = InferenceManager(
        model_name=args.model,
        use_cuda=torch.cuda.is_available() and not args.no_cuda,
        save_visualisations=not args.no_save_vis,
        save_dir=args.save_dir)
    inference_manager.predict(image_path=args.image)
