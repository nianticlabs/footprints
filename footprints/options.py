import argparse


class Options:

    def __init__(self):
        self.options = None

        self.parser = argparse.ArgumentParser()

        # Universal Options

        self.parser.add_argument('--mode',
                                 help='training or inference mode',
                                 type=str,
                                 choices=['train', 'inference'],
                                 default='train')

        self.parser.add_argument('--height',
                                 help='height of input images',
                                 type=int,
                                 default=192)
        self.parser.add_argument('--width',
                                 help='width of input images',
                                 type=int,
                                 default=640)

        self.parser.add_argument('--depth_range',
                                 help='range of depth values',
                                 nargs='+',
                                 type=float,
                                 default=[0.1, 100])

        # Training Options

        self.parser.add_argument('--training_dataset',
                                 help='dataset to train on',
                                 type=str,
                                 choices=['kitti', 'matterport'],
                                 default='kitti')

        self.parser.add_argument('--epochs',
                                 help='number of epochs to train for',
                                 type=int,
                                 default=10)
        self.parser.add_argument('--log_freq',
                                 help='sets the frequency of logs to tensorboard',
                                 type=int,
                                 default=250)

        self.parser.add_argument('--val_batches',
                                 help='number of validation batches to run and average over',
                                 type=int,
                                 default=10)

        self.parser.add_argument('--batch_size',
                                 help='number of images in each batch',
                                 type=int,
                                 default=12)

        self.parser.add_argument('--lr',
                                 help='the learning rate',
                                 type=float,
                                 default=1e-4)

        self.parser.add_argument('--use_footprint_prior',
                                 help='if set, will assume we only have positive labels for hidden '
                                      'ground, and will add a prior for pixels to not be ground',
                                 action='store_true')

        self.parser.add_argument('--footprint_prior',
                                 help='weighting to apply to negative hidden footprint labels',
                                 type=float,
                                 default=0.25)

        self.parser.add_argument('--no_depth_mask',
                                 help='if set, definitely not ground pixels will not be used',
                                 action='store_true')

        self.parser.add_argument('--moving_objects_method',
                                 help='defines how to mask moving objects - either use "ours" '
                                      '(flow and depth) or '
                                      '"none"',
                                 type=str,
                                 choices=['none', 'ours'],
                                 default='ours')
        self.parser.add_argument('--project_down_baseline',
                                 help='if set, will train a binary classifier to predict depth'
                                      'mask pixels, i.e. baseline with no reprojection',
                                 action='store_true')

        self.parser.add_argument('--num_workers',
                                 help='number of workers for dataloaders',
                                 type=int,
                                 default=8)

        self.parser.add_argument('--config_path',
                                 help='path to the json file containing dataset information',
                                 type=str,
                                 default='paths.yaml')

        self.parser.add_argument('--model_name',
                                 help='the name of the model for saving',
                                 type=str,
                                 default='model')

        self.parser.add_argument('--log_path',
                                 help='the path to save tensorboard events and trained models to',
                                 type=str,
                                 default='./logs')

        # Test Options

        self.parser.add_argument('--inference_data_type',
                                 choices=['kitti', 'matterport'],
                                 default='kitti')

        self.parser.add_argument('--load_path',
                                 help='the model path to load from',
                                 type=str)

        self.parser.add_argument('--inference_save_path',
                                 help='path to save test results to, if left as default will save '
                                      'to "load_path/<data_type>_predictions/"',
                                 default=None)

        self.parser.add_argument('--save_test_visualisations',
                                 action='store_true')

    def parse(self):
        """ Parse arguments """
        self.options = self.parser.parse_args()
        return self.options
