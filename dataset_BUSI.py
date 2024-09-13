import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import re
class BUSIDataset(Dataset):
    def __init__(self, base_path, split, busi_class, transform=None):
        self.split = split
        self.base_path = base_path
        self.transform = transform
        self.busi_class = busi_class
        
        # Define paths
        self.benign_path = os.path.join(self.base_path, 'benign')
        self.malignant_path = os.path.join(self.base_path, 'malignant')

        # Load filenames
        benign_file_names = sorted(os.listdir(self.benign_path))
        malignant_file_names = sorted(os.listdir(self.malignant_path))
        if self.busi_class == 'benign':
            self.all_files_names = benign_file_names
            self.pattern = r'benign \((\d+)\)'
        elif self.busi_class == 'malignant':
            self.all_files_names = malignant_file_names
            self.pattern = r'malignant \((\d+)\)'
        else:
            raise ValueError(f"Unknown busi_class: {self.busi_class}")

        # Prepare image-label pairs
        self.image_label_pairs = []
        for file in self.all_files_names:
            if file.endswith('.png'):
                image_path = os.path.join(self.benign_path if 'benign' in file else self.malignant_path, file)
                label_path = image_path[:-4] + '_mask.png'
                if os.path.exists(label_path):
                    self.image_label_pairs.append((image_path, label_path))

        # Shuffle and split data
        np.random.seed(42)
        np.random.shuffle(self.image_label_pairs)
        
        total_size = len(self.image_label_pairs)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        self.train_arr = self.image_label_pairs[:train_size]
        self.val_arr = self.image_label_pairs[train_size:train_size + val_size]
        self.test_arr = self.image_label_pairs[train_size + val_size:]

    def __len__(self):
        if self.split == 'train':
            return len(self.train_arr)
        elif self.split == 'val':
            return len(self.val_arr)
        elif self.split == 'test':
            return len(self.test_arr)
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __getitem__(self, index):
        if self.split == 'train':
            image_path, mask_path = self.train_arr[index]
        elif self.split == 'val':
            image_path, mask_path = self.val_arr[index]
        elif self.split == 'test':
            image_path, mask_path = self.test_arr[index]

        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        
        image = np.array(image).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        if np.max(mask)==255:
            mask/=255.0
        if np.sum(mask) == 0:
            # Handle empty masks if necessary; this is a placeholder
            return self[index + 1] if index + 1 < len(self) else (image, mask)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        filename = re.search(self.pattern, image_path).group()
        return image, mask, filename

