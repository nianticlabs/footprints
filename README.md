# Stop Crowd

Stop Crowd is an application that aims to detect and signal groups of people in a scene that are too close to each other.
This functionality could be used in public places to automatically discover gatherings and signal them to someone (e.g. law enforcement or private security).
This is important in the actual scenario to try to stop the spread of COVID-19.

This work is based on Niantic, Inc.'s footprints work _Footprints and Free Space from a Single Color Image_
([paper](https://arxiv.org/abs/2004.06376), [repository](https://github.com/nianticlabs/footprints)) and on 
Ross Wightman's PoseNet PyTorch implementation of the Google TensorFlow.js Posenet model ([repository](https://github.com/rwightman/posenet-pytorch))

## Footprints

*Footprints* is a method for estimating the visible and hidden traversable space from a single RGB image

This method predicts the hidden ground geometry and extent from a single image and estimate the depth:

<p align="center">
  <img src="readme_ims/figure_1.png" alt="Web version of figure 1" width="700" />
</p>



## ⚙️ Setup

The code and models were developed with PyTorch 1.3.1.
The `environment.yml` and `requirements.txt` list our dependencies.

We recommend installing and activating a new conda environment from these files with:
```shell
conda env create -f environment.yml -n footprints
conda activate footprints
```


## 🖼️ Prediction

We use `handheld` as a model trained on handheld stereo footage with a resolution of 256x448.

We provide code to make predictions for a single image, a whole folder of images or a video.
Model will be [automatically downloaded when required](footprints/utils.py#L105), and input images will be automatically resized to the [correct input resolution](footprints/predict_simple.py#21).

You can run the following commands from the root of the project.

Single image prediction:
```shell
python predict.py --image test_data/cyclist.jpg
```

Multi image prediction:
```shell
python predict.py --image test_data
```

By default, `.jpg` visualisations will be saved to the `predictions` folder; this can be changed with the `--save_dir` flag.

## Command line arguments

The are many optional command line arguments that allows to change the program behaviour:
- `--verbose` writes the execution time of the program and its parts to `execution_time.txt` and `execution_time.csv` in the output folder;
- `--showplt` shows a 3D graph with the keypoints of all the people recognized in the scene;
- `--no_cuda` execute all the computation on CPU. Useful if you don't have a CUDA-capable device or it doesn't have sufficient dedicated memory (4GB).
- `--more_output` useful to get additional intermediate info:
    - coordinates and scores of all the poses and keypoints found by PoseNet;
    - the original output of footprints is saved in the output folder as `<filename_without_ext>_footprints.<ext>`;
    - the depth colormap obtained from footptints is saved as above with the `depth` suffix.

## Training & more informations

The instructions on how to train the model and more info about footprints are available in the [project repository](https://github.com/nianticlabs/footprints).
