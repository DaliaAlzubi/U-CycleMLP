import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from trainer import trainer_synapse
from datetime import datetime
from trainer_ACDC import trainer_ACDC
from trainer_BUSI import trainer_BUSI
from trainer_ISIC import trainer_ISIC
from pab import DAWEU_NET

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--output_dir', type=str, help='output dir')                   
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--busi_class', type=str,
                    default=None, help='input busi class name: benign or malignant', choices=['benign', 'malignant'])

args = parser.parse_args()



if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'ACDC': {
            'root_path': '../ACDC/train',
            'list_dir': '../ACDC/lists_ACDC',
            'num_classes': 4,
            'input_channels': 1
        },
        'BUSI':{
        'root_path': '../Dataset_BUSI_with_GT',
        'num_classes':1,
        'input_channels': 1
        },
        'ISIC':{
        'root_path':'../ISIC_dataset',
        'num_classes':1,
        'input_channels': 3
        }
        
    }
    folder_name = datetime.now().strftime("%H%M_%d%m%Y")
    trainer = {'Synapse': trainer_synapse,'ACDC': trainer_ACDC, 'BUSI':trainer_BUSI, 'ISIC': trainer_ISIC}
    num_classes = dataset_config[dataset_name]['num_classes']
    num_input_channels = dataset_config[dataset_name]['input_channels']
    net=DAWEU_NET(n_channels=num_input_channels, n_classes=num_classes).to('cuda')
    trainer[dataset_name](args, net, args.output_dir)

