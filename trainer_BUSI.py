import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from medpy import metric
import pandas as pd
import torch.optim as optim
from tensorboardX import SummaryWriter
from dataset_BUSI import BUSIDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.distributed import init_process_group
def ddp_setup():
    init_process_group(backend="nccl")

def save_snapshot(model,snapshot_path):
        torch.save(model, snapshot_path)
        #print(f"Epoch {epoch} | Training snapshot saved at {snapshot_path}")

def load_snapshot(snapshot_path,model):
        #loc = f"cuda:{rank}"
        snapshot = torch.load(snapshot_path)
        model.load_state_dict(snapshot["MODEL_STATE"])
        print(f"Resuming training from saved best model weights")
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


def sigmoid_focal_loss(inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean") -> torch.Tensor:
    
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Always reduce the loss to a scalar
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "none":
        pass
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


def check_accuracy(loader, model, device="cuda"):
    intersection = 0
    union = 0
    with torch.no_grad():
        for x, y,_ in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds =model(x)
            
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()
            intersection += (preds * y).sum().item()
            union += (preds + y).sum().item()
    dice_score = (2 * intersection) / (union + 1e-8)
    return dice_score

def train_fn(loader, model, optimizer, loss_fn, scaler, device, gradient_clipping):
    loop = tqdm(loader)
    training_loss=0
    for _, (data, targets,_) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets) + sigmoid_focal_loss(predictions, targets)
            training_loss+=loss
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        scaler.step(optimizer)
        scaler.update()
        
    return training_loss


train_transforms = A.Compose(
    [
        A.Resize(width=224, height=224),  # Add resizing here
#         A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(
            mean=[0.0],
            std=[1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

val_transforms = A.Compose(
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

def trainer_BUSI(args, model, snapshot_path,nodes_snapshot_path='BUSI_current_snapshot.pth'):
    seed=42
    # savebest_model_path = os.path.join(snapshot_path, 'ACDC_best_model.pth')
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)

    # If using GPU, set seed for CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU.
        # Ensure deterministic behavior on GPU
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    
    logging.info(str(args))
    print(args)
   
    BATCH_SIZE = args.batch_size * args.n_gpu
    NUM_WORKERS = 2
    PIN_MEMORY = True

    train_ds = BUSIDataset(
    base_path='../Dataset_BUSI_with_GT',
    split='train',
    busi_class=args.busi_class,
    transform=train_transforms,
)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )

    val_ds = BUSIDataset(
        base_path='../Dataset_BUSI_with_GT',
        split='val',
        busi_class=args.busi_class,
        transform=val_transforms,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )
    print("The length of train set is: {}".format(len(train_ds)))
    print("The length of validation set is: {}".format(len(val_ds)))
    print("The length of train loader is: {}".format(len(train_loader)))
    print("The length of validation loader is: {}".format(len(val_loader)))
    LAST_EPOCH=0
    savebest_model_path=os.path.join(snapshot_path, 'best_BUSI_model.pth')
    
    if os.path.exists(savebest_model_path):
        print(f'model snapshot exists, loading pretrained weights into the model')
        model=model.load_state_dict(torch.load(savebest_model_path))
    else:
        print(f'model snapshot does not exist, training from scratch')
    model.train()
    
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
  
    LEARNING_RATE = args.base_lr
    WEIGHT_DECAY = 1e-7
    GRADIENT_CLIPPING = 1.0
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,  weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    NUM_EPOCHS = args.max_epochs

    writer = SummaryWriter(snapshot_path + '/BUSI_dataset_training_log')
    metrics_csv=[]
    best_model = None

    LAST_DICE_LOSS = float('-inf')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        training_Loss = train_fn(train_loader, model, optimizer, 
                                loss_fn, scaler, 'cuda', GRADIENT_CLIPPING)

        model.eval()
        dice_score=check_accuracy(val_loader,model, device='cuda')
        if dice_score > LAST_DICE_LOSS:
            LAST_DICE_LOSS = dice_score
            best_model = model.state_dict()
            torch.save(model.state_dict(), savebest_model_path)
            logging.info("save model to {}".format(savebest_model_path))
        else:
            model.load_state_dict(best_model)
            print(f'best model weights loaded ')
        scheduler.step(dice_score)

        LAST_EPOCH+=1
        writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('info/total_loss', training_Loss, epoch)
        # Append metrics to the list
        metrics_csv.append([epoch + 1, training_Loss.item(), dice_score])
        # Write metrics to CSV
        metrics_df = pd.DataFrame(metrics_csv, columns=['Epoch', 'Train Loss', 'Val Mean Dice'])
        metrics_df.to_csv(os.path.join(snapshot_path, 'BUSI_metrics.csv'), index=False)


        print(f'Current learning rate: {optimizer.param_groups[0]["lr"]}')
        print(f"Epoch {LAST_EPOCH} Training Loss: {training_Loss}")
        print(f"Epoch {LAST_EPOCH} Validation Metrics: ")
        print(f"Dice score: {dice_score:.4f}")
      
    writer.close()
    return 'Training Finished !!'
