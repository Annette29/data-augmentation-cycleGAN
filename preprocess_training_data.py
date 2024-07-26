import torch
from torch.utils.data import DataLoader, random_split, Dataset, RandomSampler, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from PIL import Image
import random

# Define 2 mask naming functions - pathology images are paired with corresponding masks while healthy images are paired with random pathology masks
def default_mask_name_func(image_name):
    return image_name.replace('.png', '_mask.png').replace('.jpg', '_mask.png')

def random_pathological_mask_name_func(image_name, healthy_mask_filenames):
    # Choose a random mask filename from healthy_mask_filenames
    random_mask_filename = random.choice(healthy_mask_filenames)
    return random_mask_filename

# Dataset class
class OriginalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None, mask_name_func=None, healthy_mask_filenames = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]
        self.mask_name_func = mask_name_func if mask_name_func is not None else default_mask_name_func
        self.healthy_mask_filenames = healthy_mask_filenames

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, self.mask_name_func(image_name))

        try:
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')

            if self.transform:
                image = self.transform(image)

            if self.mask_transform:
                mask = self.mask_transform(mask)

            return image, mask, image_name

        except Exception as e:
            print(f"Error loading image {image_path} or mask {mask_path}: {e}")
            return None, None, image_name

# Create dataloaders function
def create_dataloaders(image_dir, mask_dir, batch_size=8, num_workers=10, shuffle=True, pin_memory=True, mask_name_func=None, healthy_mask_filenames=None, random_sampling=False):
    if mask_name_func is None:
        mask_name_func = default_mask_name_func  # Default to using default_mask_name_func

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images to [-1, 1]
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)  # Normalize mask to [-1, 1]
    ])

    dataset = OriginalDataset(image_dir, mask_dir, transform=transform, mask_transform=mask_transform, mask_name_func=mask_name_func, healthy_mask_filenames=healthy_mask_filenames)

    if random_sampling:
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=4)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=4)

    return dataloader
