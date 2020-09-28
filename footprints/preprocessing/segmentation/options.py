import argparse


class Options:

    def __init__(self):
        self.options = None

        self.parser = argparse.ArgumentParser()

        # Universal Options

        self.parser.add_argument('--mode',
                                 help='training or testing mode',
                                 type=str,
                                 choices=['train', 'inference'],
                                 default='train')

        self.parser.add_argument('--config_path',
                                 default='paths.yaml',
                                 help='path to config file containing path information',
                                 type=str)

        self.parser.add_argument('--height',
                                 help='height of input images',
                                 type=int,
                                 default=192)
        self.parser.add_argument('--width',
                                 help='width of input images',
                                 type=int,
                                 default=640)
        self.parser.add_argument('--no_PSP',
                                 action='store_true')

        # Training Options
        self.parser.add_argument('--training_datasets',
                                 help='dataset(s) to train on',
                                 type=str,
                                 nargs='+',
                                 choices=['ADE20K', 'cityscapes', 'matterport'],
                                 default=['ADE20K', 'cityscapes'])
        self.parser.add_argument('--epochs',
                                 help='number of epochs to train for',
                                 type=int,
                                 default=20)
        self.parser.add_argument('--log_freq',
                                 help='sets the frequency of logs to tensorboard',
                                 type=int,
                                 default=250)

        self.parser.add_argument('--batch_size',
                                 help='number of images in each batch',
                                 type=int,
                                 default=12)

        self.parser.add_argument('--val_batches',
                                 help='number of validation batches to average over',
                                 type=int,
                                 default=10)

        self.parser.add_argument('--lr',
                                 help='the learning rate',
                                 type=float,
                                 default=1e-4)
        self.parser.add_argument('--num_workers',
                                 help='the number of workers for dataloading',
                                 type=int,
                                 default=4)

        self.parser.add_argument('--model_name',
                                 help='the name of the model for saving',
                                 type=str,
                                 default='model')

        self.parser.add_argument('--log_path',
                                 help='the path to save tensorboard events and trained models to',
                                 type=str,
                                 default='./logs')

        # Test Options

        self.parser.add_argument('--load_path',
                                 help='the model path to load from',
                                 type=str)

        self.parser.add_argument('--test_save_folder',
                                 help='folder to save results to - added to training_data path'
                                      'from config',
                                 type=str,
                                 default='ground_seg')

        self.parser.add_argument('--test_data_type',
                                 choices=['kitti', 'matterport'],
                                 default='kitti')

        self.parser.add_argument('--save_test_visualisations',
                                 action='store_true')

    def parse(self):
        """ Parse arguments """
        self.options = self.parser.parse_args()
        return self.options
