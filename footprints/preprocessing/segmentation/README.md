## Training

To train a ground segmentation model, you will need to download [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
and [Cityscapes](https://www.cityscapes-dataset.com/) datasets.

After downloading, modify `paths.yaml` to point to the correct paths.

To train a segmentation model, run:
```shell
CUDA_VISIBLE_DEVICES=X  python -m \ 
  footprints.preprocessing.segmentation.main
  --log_path <where_to_save>
  --model_name <my_outdoor_model>
  
```
## Inference

To generate predictions for evaluation using a trained model, run:
```shell
python -m footprints.evaluation.inference \
    --load_path <your_model_path, e.g. logs/my_outdoor_model/models/weights_19> \
    --test_data_type <KITTI or Matterport>
```
optionally specifying where to save to (defualts to the `ground_seg` folder inside the `training_data` path
specified in `paths.yaml`)