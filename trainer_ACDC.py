import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from medpy import metric
import pandas as pd
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from utils import DiceLoss
from dataset_ACDC import ACDC_dataset, RandomGenerator
from torchvision import transforms
from utils import test_single_volume
from torch.distributed import init_process_group
from torch.utils.data.distributed import DistributedSampler
def ddp_setup():
    init_process_group(backend="nccl")

def save_snapshot(model,snapshot_path):
        
        torch.save(model, snapshot_path)
        print(f"Training snapshot saved at {snapshot_path}")

def load_snapshot(snapshot_path,model):
        # loc = f"cuda:{rank}"
        snapshot = torch.load(snapshot_path)
        model.load_state_dict(snapshot)
   
        print(f"Resuming training from saved best model weights ")
        return model

def calculate_metric_per_batch(pred_batch, gt_batch):
    batch_size = pred_batch.shape[0]
    dice_scores = []
    hd95_scores = []

    for i in range(batch_size):
        pred = pred_batch[i]
        gt = gt_batch[i]
        pred[pred > 0] = 1
        gt[gt > 0] = 1

        if pred.sum() > 0 and gt.sum() > 0:
            dice = metric.binary.dc(pred, gt)
            hd95 = metric.binary.hd95(pred, gt)
            dice_scores.append(dice)
            hd95_scores.append(hd95)
        elif pred.sum() > 0 and gt.sum() == 0:
            dice_scores.append(1)
            hd95_scores.append(0)
        else:
            dice_scores.append(0)
            hd95_scores.append(0)

    mean_dice = np.mean(dice_scores)
    mean_hd95 = np.mean(hd95_scores)

    return mean_dice, mean_hd95


def trainer_ACDC(args, model, snapshot_path,nodes_snapshot_path='ACDC_current_snapshot.pth'):
   
    savebest_model_path = os.path.join(snapshot_path, 'ACDC_best_model.pth')
    # torch.backends.cudnn.benchmark = False
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # ddp_setup()
    VAL_INTERVAL=5
    epoch_run=0
    # local_rank =int(os.environ['LOCAL_RANK'])
    # global_rank=int(os.environ['RANK'])
    GRADIENT_CLIPPING = 1.0


    base_lr = args.base_lr
    # num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    print(args)
    db_train = ACDC_dataset(base_dir='../ACDC/train', list_dir='../ACDC/lists_ACDC', split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_val = ACDC_dataset(base_dir='../ACDC/train', list_dir='../ACDC/lists_ACDC', split="test")
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of validation set is: {}".format(len(db_val)))
    # def worker_init_fn(worker_id):
    #     random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,)
                             #worker_init_fn=worker_init_fn,sampler=DistributedSampler(db_train))
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=2, pin_memory=True,)
                           #worker_init_fn=worker_init_fn,sampler=DistributedSampler(db_val))
    
    print("The length of train loader is: {}".format(len(trainloader)))
    print("The length of validation loader is: {}".format(len(valloader)))
    last_dice=None
    # if args.n_gpu > 1:
    #     model = nn.DataParallel(model)
    
    # model=model.to(local_rank)
    best_model = None
    if os.path.exists(savebest_model_path):
        model=load_snapshot(savebest_model_path,model)
        best_model=model.state_dict()

    # model=DDP(model,device_ids=[local_rank])
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(4)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr)
   

    #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-7)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    writer = SummaryWriter(snapshot_path + '/ACDC_dataset_training_log')
    # Enable CUDNN benchmark for better performance
    torch.backends.cudnn.benchmark = True
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    metrics_csv=[]
    iterator = tqdm(range(epoch_run,max_epoch), ncols=70)
    
    LAST_DICE_LOSS = float('-inf')

    for epoch_num in iterator:
        model.train()
        running_loss = 0.0

        for _, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(non_blocking=True), label_batch.cuda(non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(image_batch)
                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = 0.4*loss_ce + 0.6*loss_dice
                running_loss+=loss
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
            scaler.step(optimizer)
            scaler.update()
            
            
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
        running_loss=running_loss/len(trainloader)
        logging.info(f"Training Loss: {running_loss}")
            # # Validation step
        model.eval()
        
        metric_list=0.0
        with torch.no_grad():
            for _, sampled_batch in enumerate(valloader):
                image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
                metric_i = test_single_volume(image, label, model, classes=4, patch_size=[args.img_size, args.img_size],
                                            test_save_path=None, case=case_name, z_spacing=1)
                metric_list += np.array(metric_i)
          
        metric_list = metric_list / len(db_val)
        performance = np.mean(metric_list, axis=0)[0]
        if performance > LAST_DICE_LOSS:
            LAST_DICE_LOSS = performance
            best_model=model.state_dict()
            torch.save(best_model, savebest_model_path)

        else:
            model.load_state_dict(best_model)
   
            print(f'Dice did not improve, best model weights loaded')
        
        scheduler.step(performance)
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        logging.info('Testing performance in Val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    # Append metrics to the list
        metrics_csv.append([epoch_num + 1, running_loss.item(), performance, mean_hd95])
        # Write metrics to CSV
        metrics_df = pd.DataFrame(metrics_csv, columns=['Epoch', 'Train Loss', 'Val Mean Dice', 'Val Mean HD95'])
        metrics_df.to_csv(os.path.join(snapshot_path, 'ACDC_metrics.csv'), index=False)

        if LAST_DICE_LOSS > 0.91:
            break

    

    iterator.close()
    # torch.save(best_model, savebest_model_path)
 
    writer.close()
    return "Training Finished!"

