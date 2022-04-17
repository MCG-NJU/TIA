import argparse


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')

    # About Data
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='training dataset',
        default='pascal_voc',
        type=str
    )
    parser.add_argument(
        '--nw', 
        dest='num_workers',
        help='number of worker to load data',
        default=4, 
        type=int
    )
    parser.add_argument(
        '--bs', 
        dest='batch_size',
        help='batch_size',
        default=1, 
        type=int
    )
    parser.add_argument(
        '--lamda', 
        dest='lamda',
        help='lamda',
        default=1.0, 
        type=float
    )
    parser.add_argument(
        '--lamda1', 
        dest='lamda1',
        help='lamda1',
        default=1.0, 
        type=float
    )
    parser.add_argument(
        '--lamda2', 
        dest='lamda2',
        help='lamda2',
        default=1.0, 
        type=float
    )
    parser.add_argument(
        '--lamda3', 
        dest='lamda3',
        help='lamda3',
        default=1.0, 
        type=float
    )
    parser.add_argument(
        '--alpha', 
        dest='alpha',
        help='alpha',
        default=1.0, 
        type=float
    )
    parser.add_argument(
        '--alpha1', 
        dest='alpha1',
        help='alpha1',
        default=1.0, 
        type=float
    )
    parser.add_argument(
        '--alpha2', 
        dest='alpha2',
        help='alpha2',
        default=1.0, 
        type=float
    )
    parser.add_argument(
        '--alpha3', 
        dest='alpha3',
        help='alpha3',
        default=1.0, 
        type=float
    )
    parser.add_argument(
        '--num_aux1', 
        dest='num_aux1',
        help='num_aux1',
        default=4, 
        type=int
    )
    parser.add_argument(
        '--num_aux2', 
        dest='num_aux2',
        help='num_aux2',
        default=4, 
        type=int
    )
    parser.add_argument(
        '--gamma', 
        dest='gamma',
        help='gamma',
        default=5.0, 
        type=float
    )
    parser.add_argument(
        '--warmup', 
        dest='warmup',
        help='using warming up',
        action='store_true'
    )
    parser.add_argument(
        '--context', 
        dest='context',
        help='using context-based reguralization',
        action='store_true'
    )
    parser.add_argument(
        '--preserve', 
        dest='preserve',
        help='preserving epochs',
        action='store_true'
    )
    parser.add_argument(
        '--desp', 
        dest='desp',
        help='description',
        type=str
    )

    # About Net
    parser.add_argument(
        '--net',
        dest='net',
        help='vgg16, res101',
        default='vgg16',
        type=str
    )
    parser.add_argument(
        '--ls', 
        dest='large_scale',
        help='whether use large imag scale',
        action='store_true'
    )
    parser.add_argument(
        '--cag', 
        dest='class_agnostic',
        help='whether perform class_agnostic bbox regression',
        action='store_true'
    )
    parser.add_argument(
        '--cuda', 
        dest='cuda',
        help='whether use CUDA',
        action='store_true'
    )

    # About Schedule
    parser.add_argument(
        '--s', 
        dest='session',
        help='training session',
        default=1, 
        type=int
    )    
    parser.add_argument(
        '--start_epoch',
        dest='start_epoch',
        help='starting epoch',
        default=1,
        type=int
    )
    parser.add_argument(
        '--epochs',
        dest='max_epochs',
        help='number of epochs to train',
        default=7,
        type=int
    )    
    parser.add_argument(
        "--max_iter",
        dest="max_iter",
        help="max iteration for train",
        default=10000,
        type=int,
    )

    # About Optimizor
    parser.add_argument(
        '--o',
        dest='optimizer',
        help='training optimizer',
        default="sgd", 
        type=str
    )
    parser.add_argument(
        '--lr', 
        dest='lr',
        help='starting learning rate',
        default=0.001, 
        type=float
    )
    parser.add_argument(
        '--lr_decay_step', 
        dest='lr_decay_step',
        help='step to do learning rate decay, unit is epoch',
        default=5,
        type=int
    )
    parser.add_argument(
        '--lr_decay_gamma',
        dest='lr_decay_gamma',
        help='learning rate decay ratio',
        default=0.1, 
        type=float
    )

    # About Disp
    parser.add_argument(
        '--disp_interval',
        dest='disp_interval',
        help='number of iterations to display',
        default=100,
        type=int
    )
    parser.add_argument(
        '--checkpoint_interval',
        dest='checkpoint_interval',
        help='number of iterations to display',
        default=1,
        type=int
    )

    # About States
    parser.add_argument(
        '--r', 
        dest='resume',
        help='resume checkpoint or not',
        default=False, 
        type=bool
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        help="resume from which model",
        default="",
        type=str,   
    )
    parser.add_argument(
        '--save_dir', 
        dest='save_dir',
        help='directory to save models', 
        default="models",
        type=str
    )

    args = parser.parse_args()
    return args


def parse_args_test():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')

    # About Data
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='testing dataset',
        default='pascal_voc',
        type=str
    )

    # About Net
    parser.add_argument(
        '--net',
        dest='net',
        help='vgg16, res101',
        default='vgg16',
        type=str
    )
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        help="directory to load models",
        default="models.pth",
        type=str,
    )
    parser.add_argument(
        '--ls', 
        dest='large_scale',
        help='whether use large imag scale',
        action='store_true'
    )
    parser.add_argument(
        '--cag', 
        dest='class_agnostic',
        help='whether perform class_agnostic bbox regression',
        action='store_true'
    )
    parser.add_argument(
        '--cuda', 
        dest='cuda',
        help='whether use CUDA',
        action='store_true'
    )
    parser.add_argument(
        '--context', 
        dest='context',
        help='using context-based reguralization',
        action='store_false'
    )
    parser.add_argument(
        '--num_aux1', 
        dest='num_aux1',
        help='num_aux1',
        default=4, 
        type=int
    )
    parser.add_argument(
        '--num_aux2', 
        dest='num_aux2',
        help='num_aux2',
        default=4, 
        type=int
    )

    # About Config
    parser.add_argument(
        '--cfg', 
        dest='cfg_file',
        help='optional config file',
        default='cfgs/vgg16.yml', 
        type=str
    )
    parser.add_argument(
        '--set', 
        dest='set_cfgs',
        help='set config keys', 
        default=None,
        nargs=argparse.REMAINDER
    )

    # About Visualizing
    parser.add_argument(
        '--vis', 
        dest='vis',
        help='visualization mode',
        action='store_true'
    )
    args = parser.parse_args()
    return args
