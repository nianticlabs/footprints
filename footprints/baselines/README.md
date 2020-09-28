# Footprints Baselines

To compute baselines for KITTI and Matterport datasets:

1) Follow the `Visible ground segmentation` instructions to compute visible ground segmentation

2) Follow the `3D Bounding boxes (outdoor objects)` instructions below to run bounding box predictions on KITTI

3) Follow the `3D Bounding boxes (indoor objects)` instructions below to run bounding box predictions on Matterport

4) Finally, run:

    python footprint_baselines.py --dataset kitti
    python footprint_baselines.py --dataset matterport


## 3D Bounding boxes (outdoor objects)

We compute 3D bounding boxes using [this](https://github.com/skhadem/3D-BoundingBox) implementation, which we have modified a little to process our images and create footprint predictions.

Here is how to reproduce the predictions:

1) Set up an anaconda environment with `pytorch` and `opencv >= 3.4.3`. Something like:

    conda create --name opencv__3.4.3 python=3.6
    conda activate opencv__3.4.3
    conda install -c conda-forge opencv=3.4.3
    conda install pytorch torchvision -c pytorch  # see https://pytorch.org/ for full options

2) Clone the repo from https://github.com/skhadem/3D-BoundingBox, and checkout git commit `0bf17ec`.

3) Apply the patch file `3D-BoundingBox.diff` which is contained in this directory

4) Run predictions with:

    python Run.py --image-dir path/to/images --save-dir where/to/save/predictions

5) If you modify the `3D-BoundingBox` repo and want to update the patch file: have a look at `create_patch.sh` which will have been added to the 3D-BoundingBox repo after you have applied the `3D-BoundingBox.diff` patch (a bit circular I know...).



## 3D Bounding boxes (indoor objects)

We compute 3D bounding boxes using [this](https://github.com/facebookresearch/votenet) method, which only uses point clouds as input (yet still gets SOTA results, they say).
There are two models, one trained on scannet and one on sunrgbd; we are using each model as a separate baseline, at least for now.

Here is how to reproduce the predictions:

1) Clone the repo from https://github.com/facebookresearch/votenet, and checkout git commit `257b8d8`.

2) Follow their setup instructions on their README

3) Apply the patch file `votenet.diff` which is contained in this directory

4) Run predictions with:

    python matterport_predict.py scannet

    or

    python matterport_predict.py sunrgbd

5) As for `3D-BoundingBox`: If you modify the `votenet` repo and want to update the patch file: have a look at `create_patch.sh` which will have been added to the votenet repo after you have applied the `votenet.diff` patch (a bit circular I know...).
