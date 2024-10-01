import os
import random
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision.transforms import functional as Ft
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = np.array(img)
        augmented = self.transform(image=img)
        return augmented['image']

albumentations_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.5),
    A.GridDistortion(p=0.5),
    A.ElasticTransform(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Apply the albumentations transform
transform = AlbumentationsTransform(albumentations_transform)

def load_dataset(directory, transform):
    dataset = []
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                file_path = os.path.join(root, file)
                img = Image.open(file_path)

                img = transform(img)
                img = Ft.resize(img, (224, 224))

                if not isinstance(img, torch.Tensor):
                    img = Ft.to_tensor(img)

                dataset.append((img, 0 if 'Without Lesions' in directory else 1))
                count += 1
    return dataset, count

def sample_dataset(dataset, num_samples):
    indices = random.sample(range(len(dataset)), num_samples)
    return Subset(dataset, indices)

def load_all_datasets(base_dir):
    paths = {
        'train_real_with_lesions': f'{base_dir}/WSI Patches/Resized With Lesions/Training Data',
        'val_real_with_lesions': f'{base_dir}/WSI Patches/Resized With Lesions/Validation Data',
        'test_real_with_lesions': f'{base_dir}/WSI Patches/Resized With Lesions/Test Data',
        'train_real_without_lesions': f'{base_dir}/WSI Patches/Without Lesions/Training Data',
        'val_real_without_lesions': f'{base_dir}/WSI Patches/Without Lesions/Validation Data',
        'test_real_without_lesions': f'{base_dir}/WSI Patches/Without Lesions/Test Data',
        'train_synthetic_with_masks_with_lesions': f'{base_dir}/Fake With Masks/With Lesions/Training Data',
        'val_synthetic_with_masks_with_lesions': f'{base_dir}/Fake With Masks/With Lesions/Validation Data',
        'test_synthetic_with_masks_with_lesions': f'{base_dir}/Fake With Masks/With Lesions/Test Data',
        'train_synthetic_with_masks_without_lesions': f'{base_dir}/Fake With Masks/Without Lesions/Training Data',
        'val_synthetic_with_masks_without_lesions': f'{base_dir}/Fake With Masks/Without Lesions/Validation Data',
        'test_synthetic_with_masks_without_lesions': f'{base_dir}/Fake With Masks/Without Lesions/Test Data'
    }

    datasets = {}
    counts = {}

    for key, path in paths.items():
        datasets[key], counts[key] = load_dataset(path, transform)

    return datasets, counts

def combine_datasets(datasets, counts):
    def min_counts(keys):
        return min(counts[key] for key in keys)

    # Real + Synthetic
    train_with_masks_keys = ['train_real_with_lesions', 'train_synthetic_with_masks_with_lesions']
    val_with_masks_keys = ['val_real_with_lesions', 'val_synthetic_with_masks_with_lesions']
    test_with_masks_keys = ['test_real_with_lesions', 'test_synthetic_with_masks_with_lesions']

    min_train_with_masks = min_counts(train_with_masks_keys)
    min_val_with_masks = min_counts(val_with_masks_keys)
    min_test_with_masks = min_counts(test_with_masks_keys)

    train_combined_with_masks = ConcatDataset([
        sample_dataset(datasets[key], min_train_with_masks) for key in train_with_masks_keys
    ])
    val_combined_with_masks = ConcatDataset([
        sample_dataset(datasets[key], min_val_with_masks) for key in val_with_masks_keys
    ])
    test_combined_with_masks = ConcatDataset([
        sample_dataset(datasets[key], min_test_with_masks) for key in test_with_masks_keys
    ])

    return {
        'train_combined_with_masks': train_combined_with_masks,
        'val_combined_with_masks': val_combined_with_masks,
        'test_combined_with_masks': test_combined_with_masks
    }
