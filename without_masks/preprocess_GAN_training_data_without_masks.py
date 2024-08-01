import torch
from torch.utils.data import DataLoader, random_split, Dataset, RandomSampler, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from PIL import Image
import random

class DatasetNoMasks(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)

        try:
            image = Image.open(image_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, image_name

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, image_name

def create_dataloaders_no_masks(image_dir, batch_size=8, num_workers=10, shuffle=True, pin_memory=True, random_sampling=False):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images to [-1, 1]
    ])

    dataset = DatasetNoMasks(image_dir, transform=transform)

    if random_sampling:
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=4)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=4)

    return dataloader


