import torch
dict=torch.load("./pretrained_ckpt/imagenet_epoch127.pth", map_location="cuda")
print(dict.keys())