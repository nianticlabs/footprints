# [Footprints and Free Space from a Single Color Image](https://arxiv.org/abs/2004.06376)

**[Jamie Watson](https://scholar.google.com/citations?view_op=list_works&hl=en&user=5pC7fw8AAAAJ), [Michael Firman](http://www.michaelfirman.co.uk), [Aron Monszpart](http://aron.monszp.art) and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/) – CVPR 2020 (Oral presentation)**

[[Link to Paper](https://arxiv.org/abs/2004.06376)]


**We introduce *Footprints*, a method for estimating the visible and hidden traversable space from a single RGB image**


<p align="center">
  <a href="https://storage.googleapis.com/niantic-lon-static/research/footprints/Main_with_yeti_h264.mp4">
  <img src="readme_ims/video_title.jpg" alt="5 minute CVPR presentation video link" width="400">
  </a>
</p>

Understanding the shape of a scene from a single color image is a formidable computer vision task.
Most methods aim to predict the geometry of surfaces that are visible to the camera, which is of limited use when planning paths for robots or augmented reality agents. Models which predict beyond the line of sight often parameterize the scene with voxels or meshes, which can be expensive to use in machine learning frameworks.

Our method predicts the hidden ground geometry and extent from a single image:

<p align="center">
  <img src="readme_ims/figure_1.png" alt="Web version of figure 1" width="700" />
</p>

Our predictions enable virtual characters to more realistically explore their environment.

<table width="700" align="center">
  <tr>
    <td><img src="readme_ims/penguin_baseline.gif" alt="Baseline exploration" width="300" /></td>
    <td><img src="readme_ims/penguin_ours.gif" alt="Our exploration" width="300" /></td>
  </tr>
  <tr>
    <td><b>Baseline:</b> The virtual character can only explore the ground visible to the camera</td>
    <td><b>Ours:</b> The penguin can explore both the visible and hidden ground</td>
  </tr>
</table>



## ⚙️ Setup

Our code and models were developed with PyTorch 1.3.1.
The `environment.yml` and `requirements.txt` list our dependencies.

We recommend installing and activating a new conda environment from these files with:
```shell
conda env create -f environment.yml -n footprints
conda activate footprints
```


## 🖼️ Prediction

We provide three pretrained models:

- `kitti`, a model trained on the KITTI driving dataset with a resolution of 192x640,
- `matterport`, a model trained on the indoor Matterport dataset with a resolution of 512x640, and
- `handheld`, a model trained on our own handheld stereo footage with a resolution of 256x448.

We provide code to make predictions for a single image, or a whole folder of images, using any of these pretrained models.
Models will be [automatically downloaded when required](footprints/utils.py#L35), and input images will be automatically resized to the [correct input resolution](footprints/predict.py#21) for each model.

Single image prediction:
```shell
python -m footprints.predict --image test_data/cyclist.jpg --model kitti
```

Multi image prediction:
```shell
python -m footprints.predict --image test_data --model handheld
```

By default, `.npy` predictions and `.jpg` visualisations will be saved to the `predictions` folder; this can be changed with the `--save_dir` flag.


*Training code is coming soon*


## ⏳ Evaluation

To evaluate a folder of predictions, run:
```shell
python -m footprints.evaluate \
    --datatype kitti \
    --metric iou \
    --predictions path/to/predictions/folder
```

The following options are provided:
- `--datatype` can be either `kitti` or `matterport`.
- `--metric` can be `iou` or `depth`

If necessary, the ground truth files will be automatically downloaded and placed in the `ground_truth_files` folder.

You can also download the KITTI annotations directly from [here](https://storage.googleapis.com/niantic-lon-static/research/footprints/data/kitti/kitti_ground_truth.zip).
For each image, there are 3 `.png` files:

- `XXXXX_ground.png` contains the mask of the boundary of visible and hidden ground, ignoring all objects
- `XXXXX_objects.png` contains the mask of the ground space taken up by objects (the *footprints*)
- `XXXXX_combined.png` contains the full evaluation mask - the visible and hidden ground, taking into account object footprints

E.g. evaluating on the KITTI test set (assuming images are in a folder named `KITTI_test_rgbs`) could be done by:
```shell
python -m footprints.predict \
 --image KITTI_test_rgbs \
 --model kitti \
 --save_dir ./predictions
python -m footprints.evaluate \
 --datatype kitti \
 --metric iou \
 --predictions ./predictions/outputs
```

## Method and further results

We learn from stereo video sequences, using camera poses, per-frame depth and semantic segmentation to form training data, which is used to supervise an image-to-image network.

<p align="center">
  <img src="readme_ims/figure_3.gif" alt="Video version of figure 3" width="900" />
</p>

Results on mobile phone footage:

<table width="700" align="center">
  <tr>
    <td><img src="readme_ims/ours_1.gif" alt="Rig results" width="300" /></td>
    <td><img src="readme_ims/ours_2.gif" alt="Rig results" width="300" /></td>
  </tr>
</table>

More results on the KITTI dataset:
<p align="center">
  <img src="readme_ims/kitti_results.gif" alt="KITTI results" width="600" />
</p>


## ✏️ 📄 Citation

If you find our work useful or interesting, please consider citing [our paper](https://arxiv.org/abs/2004.06376):

```
@inproceedings{watson-2020-footprints,
 title   = {Footprints and Free Space from a Single Color Image},
 author  = {Jamie Watson and
            Michael Firman and
            Aron Monszpart and
            Gabriel J. Brostow},
 booktitle = {Computer Vision and Pattern Recognition ({CVPR})},
 year = {2020}
}
```


# 👩‍⚖️ License
Copyright © Niantic, Inc. 2020. Patent Pending. All rights reserved. Please see the license file for terms.
