import torch
import torch.optim as optim
import torch.autograd as autograd
from torch.nn.utils import clip_grad_norm_
import torch.nn.parallel
from torch.autograd import Variable
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import gc
import itertools
from tempfile import TemporaryDirectory

from without_masks.preprocess_GAN_training_data_without_masks import create_dataloaders_no_masks

def initialize_components(device):
    # Create dataloaders for the training, validation, and test datasets for images with and without lesions
    batch_size = 32
    num_workers = 10
    # Define root directories
    base_dir = 'your/patches for svs images both with and without lesions/folder'
    
    # Without Lesions
    train_image_dir_healthy = os.path.join(base_dir, 'Without Lesions/Training Data')
    train_loader_healthy_no_masks = create_dataloaders_no_masks(train_image_dir_healthy)
    
    val_image_dir_healthy = os.path.join(base_dir, 'Without Lesions/Validation Data')
    val_loader_healthy_no_masks = create_dataloaders_no_masks(val_image_dir_healthy, shuffle=False, random_sampling=True)
    
    test_image_dir_healthy = os.path.join(base_dir, 'Without Lesions/Test Data')
    test_loader_healthy_no_masks = create_dataloaders_no_masks(test_image_dir_healthy, shuffle=False, random_sampling=True)
    
    # With Lesions
    train_image_dir_pathological = os.path.join(base_dir, 'Resized With Lesions/Training Data')
    train_loader_pathological_no_masks = create_dataloaders_no_masks(os.path.join(base_dir, 'Resized With Lesions/Training Data'))
    val_image_dir_pathological = os.path.join(base_dir, 'Resized With Lesions/Validation Data')
    val_loader_pathological_no_masks = create_dataloaders_no_masks(os.path.join(base_dir, 'Resized With Lesions/Validation Data'), shuffle=False, random_sampling=True)
    test_image_dir_pathological = os.path.join(base_dir, 'Resized With Lesions/Test Data')
    test_loader_pathological_no_masks = create_dataloaders_no_masks(os.path.join(base_dir, 'Resized With Lesions/Test Data'), shuffle=False, random_sampling=True)

# Path to save model checkpoints
checkpoint_path_no_masks = '/store/model checkpoints here/'
os.makedirs(checkpoint_path_no_masks, exist_ok=True)
