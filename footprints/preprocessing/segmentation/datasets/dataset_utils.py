# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from PIL import Image
import numpy as np
import random
from torchvision import transforms
import cv2


def pil_loader(path):
    """
    Load image from path with PIL. PIL is used to avoid
    ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    """
    with open(path, 'rb') as file_handler:
        with Image.open(file_handler) as img:
            return img.convert('RGB')


def prepare_size(image, labels, feed_height, feed_width, keep_aspect_ratio=True):

    height, width, _ = np.array(image).shape

    if keep_aspect_ratio:
        if feed_height <= height and feed_width <= width:
            # can simply crop the image
            target_height = height
            target_width = width

        else:
            # check the constraint
            current_ratio = height / width
            target_ratio = feed_height / feed_width

            if current_ratio < target_ratio:
                # height is the constraint
                target_height = feed_height
                target_width = int(feed_height / height * width)

            elif current_ratio > target_ratio:
                # width is the constraint
                target_height = int(feed_width / width * height)
                target_width = feed_width

            else:
                # ratio is the same - just resize
                target_height = feed_width
                target_width = feed_width

    else:
        target_height = feed_width
        target_width = feed_width

    image, labels = resize_all(image, labels, target_height, target_width)

    # now do cropping
    if target_height == feed_height and target_width == feed_width:
        # we are already at the correct size - no cropping
        pass
    else:
        image, labels = crop_all(image, labels, feed_height, feed_width)

    return image, labels


def crop_all(image, labels, feed_height, feed_width):

    # get crop parameters
    height, width, _ = np.array(image).shape
    top = int(random.random() * (height - feed_height))
    left = int(random.random() * (width - feed_width))
    right, bottom = left + feed_width, top + feed_height

    image = image.crop((left, top, right, bottom))
    labels = labels.crop((left, top, right, bottom))

    return image, labels


def resize_all(image, labels, height, width):

    label_resizer = transforms.Resize(size=(height, width), interpolation=Image.NEAREST)
    img_resizer = transforms.Resize(size=(height, width), interpolation=Image.LANCZOS)
    image = img_resizer(image)
    labels = label_resizer(labels)

    return image, labels