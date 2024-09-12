import argparse
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
from torchvision import transforms
from utils import test_single_volume
from torch.distributed import init_process_group
from torch.utils.data.distributed import DistributedSampler
def ddp_setup():
    init_process_group(backend="nccl")

def save_snapshot(epoch,model,snapshot_path):
        snapshot = {
            "MODEL_STATE": model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, snapshot_path)
        #print(f"Epoch {epoch} | Training snapshot saved at {snapshot_path}")

def load_snapshot(snapshot_path,rank,model):
        loc = f"cuda:{rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        model.load_state_dict(snapshot["MODEL_STATE"])
        epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {epochs_run}")
        return model,epochs_run
import numpy as np


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


def trainer_synapse(args, model, snapshot_path,nodes_snapshot_path='current_snapshot.pth'):
    from dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    ddp_setup()
    VAL_INTERVAL=5
    local_rank =int(os.environ['LOCAL_RANK'])
    global_rank=int(os.environ['RANK'])
    current_epoch=0

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_val = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="test_vol")
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of validation set is: {}".format(len(db_val)))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn,sampler=DistributedSampler(db_train))
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
                           worker_init_fn=worker_init_fn,sampler=DistributedSampler(db_val))
    
    print("The length of train loader is: {}".format(len(trainloader)))
    print("The length of validation loader is: {}".format(len(valloader)))

    # if args.n_gpu > 1:
    #     model = nn.DataParallel(model)
    
    model=model.to(local_rank)
    if os.path.exists(nodes_snapshot_path):
        model,current_epoch=load_snapshot(nodes_snapshot_path,local_rank,model)
    model=DDP(model,device_ids=[local_rank])
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    writer = SummaryWriter(snapshot_path + '/log')
    # Enable CUDNN benchmark for better performance
    torch.backends.cudnn.benchmark = True
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = float('inf')
    metrics_csv=[]
    iterator = tqdm(range(current_epoch,max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        running_hd95 = 0.0

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(local_rank, non_blocking=True), label_batch.cuda(local_rank, non_blocking=True)
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            running_loss+=loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            #logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            # if iter_num % 20 == 0:
            #     image = image_batch[1, 0:1, :, :]
            #     image = (image - image.min()) / (image.max() - image.min())
            #     writer.add_image('train/Image', image, iter_num)
            #     outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
            #     labs = label_batch[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/GroundTruth', labs, iter_num)
        running_loss=running_loss/len(trainloader)
        logging.info(f"Training Loss: {running_loss}")
        if (epoch_num+1) % VAL_INTERVAL == 0:
            # # Validation step
            model.eval()
            val_loss = 0.0
            
            metric_list=0.0
            with torch.no_grad():
                for i_batch, sampled_batch in enumerate(valloader):
                    image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
                    metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                                test_save_path=None, case=case_name, z_spacing=1)
                    metric_list += np.array(metric_i)
                    
                metric_list = metric_list / len(db_val)
                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                logging.info('Testing performance in Val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
                


            if (global_rank==0) and (local_rank==0):
                # Save the best model
                if val_loss < best_performance:
                    best_performance = val_loss
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.module.state_dict(), save_mode_path)
                    #logging.info("save best model to {}".format(save_mode_path))

                save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num+1) + '.pth')
                torch.save(model.module.state_dict(), save_mode_path)
                #logging.info("save model to {}".format(save_mode_path))
                # Append metrics to the list
                metrics_csv.append([epoch_num + 1, running_loss.item(), performance, mean_hd95])
                # Write metrics to CSV
                metrics_df = pd.DataFrame(metrics_csv, columns=['Epoch', 'Train Loss', 'Val Mean Dice', 'Val Mean HD95'])
                metrics_df.to_csv(os.path.join(snapshot_path, 'metrics.csv'), index=False)

        save_snapshot(epoch_num, model, nodes_snapshot_path)
    
    save_mode_path = os.path.join(snapshot_path, 'final_epoch_' + str(epoch_num+1) + '.pth')
    torch.save(model.module.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    iterator.close()
    

    writer.close()
    return "Training Finished!"
