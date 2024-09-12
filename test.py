import argparse
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from dataset_ACDC import ACDC_dataset
from dataset_BUSI import BUSIDataset
from dataset_ISIC import ISICDataset
from pab import DAWEU_NET
from scipy.ndimage import zoom
import torch
print(torch.cuda.is_available())

def save_collages(images, masks, predictions, output_dir, case_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_images = masks.shape[0]
   
    for i in range(num_images):
        # Create a figure with 1 row and 3 columns
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Image
        axes[0].imshow(images[i], cmap='gray')
        axes[0].set_title('Image')
        axes[0].axis('off')
        
        # Ground Truth Mask
        axes[1].imshow(masks[i], cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        # Predicted Mask
        axes[2].imshow(predictions[i], cmap='gray')
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')
        
        # Save the collage
        if case_name is not None:
            collage_path = os.path.join(output_dir, f'{case_name}_collage_{i}.png')
            plt.savefig(collage_path, bbox_inches='tight')
            plt.close(fig)
        else:
            collage_path = os.path.join(output_dir, f'collage_{i}.png')
            plt.savefig(collage_path, bbox_inches='tight')
            plt.close(fig)
def save_isic_collages(images, masks, predictions, output_dir, case_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_images = masks.shape[0]

    for i in range(num_images):
        # Create a figure with 1 row and 3 columns
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Image
        axes[0].imshow(np.transpose(images, (1,2,0)))
        axes[0].set_title('Image')
        axes[0].axis('off')
        
        # Ground Truth Mask
        axes[1].imshow(masks[i], cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        # Predicted Mask
        axes[2].imshow(predictions[i], cmap='gray')
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')
        
        # Save the collage
        if case_name is not None:
            collage_path = os.path.join(output_dir, f'{case_name}_collage_{i}.png')
            plt.savefig(collage_path, bbox_inches='tight')
            plt.close(fig)
        else:
            collage_path = os.path.join(output_dir, f'collage_{i}.png')
            plt.savefig(collage_path, bbox_inches='tight')
            plt.close(fig)



parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='ACDC', help='experiment_name')
parser.add_argument('--pretrained_model_path', type=str,
                    default='./output/ACDC_best_model.pth', help='model path')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', type=str, help='output dir')   
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
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

args = parser.parse_args()
#config = get_config(args)

def test_single_volume(image, label, net, patch_size=[224, 224], test_save_path=None, case_name=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
        if test_save_path is not None:
            save_collages(image, label, prediction, test_save_path, case_name)
    
   
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
    test_transforms = A.Compose(
    [
        A.Resize(width=224, height=224),  # Add resizing here
        A.Normalize(
            mean=[0.0],
            std=[1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    # args.volume_path = dataset_config[dataset_name]['volume_path']
    print(args.dataset)
    #args.dataset = dataset_config[dataset_name]['Dataset']
    if 'list_dir' in dataset_config[dataset_name].keys():
        args.list_dir = dataset_config[dataset_name]['list_dir']
    # args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True
    INPUT_CHANNLES = dataset_config[dataset_name]['input_channels']


    #define the model
    net=DAWEU_NET(n_channels=INPUT_CHANNLES, n_classes=args.num_classes).to('cuda')  
    net.eval()
    #define the snapshot path and test dataset
    if args.dataset == "ACDC":
        RESULTS_FLDER_PATH = os.path.join(args.output_dir, 'ACDC_TEST_OUTPUTS/')
        os.makedirs(RESULTS_FLDER_PATH, exist_ok=True)
        snapshot = os.path.join(args.output_dir, args.pretrained_model_path)
        db_test = ACDC_dataset(base_dir='../ACDC/train', list_dir='../ACDC/lists_ACDC', split="test")
        print(len(db_test))
        #    load the pre-trained model
        net.load_state_dict(torch.load(snapshot))
        print(f"pre-trained model loaded from {snapshot} for evalutation on {args.dataset} dataset")
        test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=2, pin_memory=True,)
        print(f"Testing on {args.dataset} dataset with {len(test_loader)} batches")
        with torch.no_grad():
            for _, sampled_batch in enumerate(test_loader):
                image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
                test_single_volume(image, label, net, patch_size=[224, 224], 
                                   test_save_path=RESULTS_FLDER_PATH, case_name=case_name)

            
        
    elif args.dataset == "BUSI":
        RESULTS_FLDER_PATH = os.path.join(args.output_dir, 'BUSI_TEST_OUTPUTS/')
        os.makedirs(RESULTS_FLDER_PATH, exist_ok=True)
        snapshot = os.path.join(args.output_dir, args.pretrained_model_path)        

        net.load_state_dict(torch.load(snapshot))
        print(f"pre-trained model loaded from {snapshot} for evalutation on {args.dataset} dataset")

        test_ds = BUSIDataset(
        base_path='../Dataset_BUSI_with_GT',
        split='test',
        transform=test_transforms,
    )
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            shuffle=False,
        )
        print(f'length of BUSI test dataset is {len(test_ds)}')
        print(f"Testing on {args.dataset} dataset with {len(test_loader)} batches")
        net.eval()
        with torch.no_grad():
            for x, y, z in test_loader:
                x = x.to('cuda')
                y = y.to('cuda').unsqueeze(1)
                preds = torch.sigmoid(net(x))
                preds = (preds > 0.5).float()
                if RESULTS_FLDER_PATH is not None:
                    save_collages(x.cpu().squeeze(0), y.cpu().squeeze(0), 
                                  preds.cpu().squeeze(0), RESULTS_FLDER_PATH, z[0])

    elif args.dataset == "ISIC":
        RESULTS_FLDER_PATH = os.path.join(args.output_dir, 'ISIC_TEST_OUTPUTS/')
        os.makedirs(RESULTS_FLDER_PATH, exist_ok=True)
        snapshot = os.path.join(args.output_dir, args.pretrained_model_path)
        
        net.load_state_dict(torch.load(snapshot))
        print(f"pre-trained model loaded from {snapshot} for evalutation on {args.dataset} dataset")

        test_ds = ISICDataset(
        base_path='../ISIC_dataset',
        split='test',
        transform=test_transforms,
    )

        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            shuffle=False,
        )
        print(f'length of ISIC test dataset is {len(test_ds)}')
        print(f"Testing on {args.dataset} dataset with {len(test_loader)} batches")
        net.eval()
        with torch.no_grad():
            for x, y, z in test_loader:
                x = x.to('cuda')
              
                y = y.to('cuda').unsqueeze(1)
                preds = torch.sigmoid(net(x))
                preds = (preds > 0.5).float()
                if RESULTS_FLDER_PATH is not None:
                    save_isic_collages(x.cpu().squeeze(0), y.cpu().squeeze(0), 
                                  preds.cpu().squeeze(0), RESULTS_FLDER_PATH, z[0])


    else:
        print('Please specify a correct dataset')
        print('Please specify correct snapshot path')
 

