import os
import re
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

class ISICDataset(Dataset):
    def __init__(self, base_path, split, transform=None):
        self.split = split
        self.base_path = base_path
        self.transform = transform
        self.pattern = r'ISIC_\d+'

        # Define paths
        if self.split=='train':
            self.images_folder = os.path.join(self.base_path, 'ISBI2016_ISIC_Part1_Training_Data/')
            self.labels_folder = os.path.join(self.base_path, 'ISBI2016_ISIC_Part1_Training_GroundTruth')
        else:
            self.images_folder = os.path.join(self.base_path, 'ISBI2016_ISIC_Part1_Test_Data')
            self.labels_folder = os.path.join(self.base_path, 'ISBI2016_ISIC_Part1_Test_GroundTruth')

        # Load filenames
        self.images_filenames = sorted(os.listdir(self.images_folder))
        self.labels_filenames = sorted(os.listdir(self.labels_folder))


    def __len__(self):
        return len(self.images_filenames)
    

    def __getitem__(self, index):
        img_path = os.path.join(self.images_folder, self.images_filenames[index])
        mask_path = os.path.join(self.labels_folder, self.labels_filenames[index])
        image=Image.open(img_path).convert('RGB')
       
        mask = Image.open(mask_path).convert('L')
        image = np.array(image).astype(np.float32)#[:,:,0]
        mask = np.array(mask).astype(np.float32)
        if np.max(mask) == 255:
            mask/=255.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        filename = re.search(self.pattern, img_path).group()
        return image, mask, filename
