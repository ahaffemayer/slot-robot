import os
from pathlib import Path
import random
import json
import numpy as np
from PIL import Image
import torch
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

class PARTNET(Dataset):
    def __init__(self, split='train'):
        super(PARTNET, self).__init__()
        
        assert split in ['train', 'val', 'test']
        self.split = split
        self.root_dir = Path(Path.cwd()) / "generated_scenes/"
        self.files = os.listdir(self.root_dir)
        self.img_transform = transforms.Compose([
               transforms.ToTensor()])

    
        # Only include .png files
        self.files = [f for f in os.listdir(self.root_dir) if f.endswith(".png")]
        self.files.sort()  # optional: for consistent ordering

        self.img_transform = transforms.Compose([
            transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])


    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.files[index])
        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)

        sample = {'image': image}
        return sample

    def __len__(self):
        return len(self.files)

class PartnetDataModule(LightningDataModule):
    def __init__(self, batch_size=16, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Load all datasets so we can access them in any stage
        self.train_set = PARTNET(split='train')
        self.val_set = PARTNET(split='val')
        self.test_set = PARTNET(split='test')

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )