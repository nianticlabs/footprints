# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


from .kitti_dataset import KITTIDataset
from .matterport_dataset import MatterportDataset
from .inference_dataset import KITTIInferenceDataset, MatterportInferenceDataset


def get_dataset_class(dataset_name):
    """
    Helper function which returns class corresponding to a dataset name
    """
    return {
        "kitti": KITTIDataset,
        "matterport": MatterportDataset
        }[dataset_name]


def get_inference_dataset_class(dataset_name):
    """
    Helper function which returns class corresponding to a dataset name
    """
    return {
        "kitti": KITTIInferenceDataset,
        "matterport": MatterportInferenceDataset
        }[dataset_name]
