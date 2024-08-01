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
from without_masks.model_without_masks_architectures import UNetResNet34, PatchGANDiscriminator, weights_init_normal, WassersteinLossGP, CombinedL1L2Loss

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

    # Initialize the 4 models for use - generator_H2P, generator_p2H, discriminator_H, and disciminator_P & the 3 loss functions to train them - WGAN-GP as the adversarial loss. identity loss, and cycle consistency loss
    generator_H2P = UNetResNet34(in_channels=3, out_channels=3).to(device)  
    generator_P2H = UNetResNet34(in_channels=3, out_channels=3).to(device)  
    discriminator_H = PatchGANDiscriminator(in_channels=3).to(device)
    discriminator_P = PatchGANDiscriminator(in_channels=3).to(device)
    
    generator_H2P.apply(weights_init_normal)
    generator_P2H.apply(weights_init_normal)
    discriminator_H.apply(weights_init_normal)
    discriminator_P.apply(weights_init_normal)
    
    wgan_gp_loss = WassersteinLossGP(lambda_gp=10)
    smooth_real_label = 0.9
    smooth_fake_label = 0.1
    criterion_cycle = CombinedL1L2Loss(lambda_l1=1.0, lambda_l2=1.0).to(device)
    criterion_identity = CombinedL1L2Loss(lambda_l1=1.0, lambda_l2=1.0).to(device)
    
    # Initialize optimizers with reducing learning rate and weight decay/L2 regularization
    optimizer_G = torch.optim.Adam(itertools.chain(generator_H2P.parameters(), generator_P2H.parameters()), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.001)
    optimizer_D_H = torch.optim.Adam(discriminator_H.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.001)
    optimizer_D_P = torch.optim.Adam(discriminator_P.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.001)
    
    # Wrap models with DataParallel for data parallelism (allows you to split the batch of data across multiple GPUs and compute the results in parallel)
    generator_H2P = torch.nn.DataParallel(generator_H2P)
    generator_P2H = torch.nn.DataParallel(generator_P2H)
    discriminator_H = torch.nn.DataParallel(discriminator_H)
    discriminator_P = torch.nn.DataParallel(discriminator_P)

    # Define the validation function
    def validate(generator_H2P, generator_P2H, val_loader_healthy_no_masks, val_loader_pathological_no_masks, criterion_cycle, criterion_identity):
        generator_H2P.eval()
        generator_P2H.eval()
    
        total_loss_id_A = 0.0
        total_loss_id_B = 0.0
        total_loss_cycle_ABA = 0.0
        total_loss_cycle_BAB = 0.0
        total_samples = 0
    
        # Initialize iterators
        pathological_iter = iter(val_loader_pathological_no_masks)
        healthy_iter = iter(val_loader_healthy_no_masks)
    
        with torch.no_grad():
            for _ in range(max(len(val_loader_pathological_no_masks), len(val_loader_healthy_no_masks))):
                try:
                    # Fetch pathological data
                    real_B, image_name_B = next(pathological_iter)
                except StopIteration:
                    # Reset iterator if StopIteration occurs
                    pathological_iter = iter(val_loader_pathological_no_masks)
                    real_B, image_name_B = next(pathological_iter)
    
                real_B = real_B.to(device)
    
                try:
                    # Fetch healthy data
                    real_A, image_name_A = next(healthy_iter)
                except StopIteration:
                    # Reset iterator if StopIteration occurs
                    healthy_iter = iter(val_loader_healthy_no_masks)
                    real_A, image_name_A = next(healthy_iter)
    
                real_A = real_A.to(device)
    
                # Ensure consistent batch sizes
                if real_A.size(0) != real_B.size(0):
                    continue
    
                # 1. Identity Loss
                loss_id_A = criterion_identity(generator_P2H(real_A), real_A)
                loss_id_B = criterion_identity(generator_H2P(real_B), real_B)
    
                # 2. Cycle Consistency Loss
                fake_B = generator_H2P(real_A.detach()).clone()
                recov_A = generator_P2H(fake_B).clone()
                loss_cycle_ABA = criterion_cycle(recov_A, real_A)
    
                fake_A = generator_P2H(real_B.detach()).clone()
                recov_B = generator_H2P(fake_A).clone()
                loss_cycle_BAB = criterion_cycle(recov_B, real_B.detach())
    
                # Accumulate total losses
                batch_size_A = real_A.size(0)
                batch_size_B = real_B.size(0)
                total_loss_id_A += loss_id_A.item() * batch_size_A
                total_loss_id_B += loss_id_B.item() * batch_size_B
                total_loss_cycle_ABA += loss_cycle_ABA.item() * batch_size_A
                total_loss_cycle_BAB += loss_cycle_BAB.item() * batch_size_B
                total_samples += batch_size_A + batch_size_B  # Counting batches processed
    
        # Calculate average losses
        avg_loss_id_A = total_loss_id_A / total_samples
        avg_loss_id_B = total_loss_id_B / total_samples
        avg_loss_cycle_ABA = total_loss_cycle_ABA / total_samples
        avg_loss_cycle_BAB = total_loss_cycle_BAB / total_samples
    
        generator_H2P.train()
        generator_P2H.train()
    
        return avg_loss_id_A, avg_loss_id_B, avg_loss_cycle_ABA, avg_loss_cycle_BAB
        
    return (
        generator_H2P, generator_P2H, discriminator_H, discriminator_P,
        train_loader_healthy_no_masks, train_loader_pathological_no_masks, val_loader_healthy_no_masks, val_loader_pathological_no_masks,
        optimizer_G, optimizer_D_H, optimizer_D_P,
        scheduler_G, scheduler_D_H, scheduler_D_P,
        criterion_identity, criterion_cycle, 
        wgan_gp_loss
    )

# Define constants and hyperparameters
epoch_counter = 0
num_epochs = 1001 # Start by training for 1000 epochs and observe the resulting outputs - we are using a while loop during training hence the +1
lambda_cycle = 10.0
lambda_identity = 10.0
sample_interval = 250  # Adjusted for less frequent loss calculation
save_interval = 200  # Save model states every 200 epochs
checkpoint_path_no_masks = '/store/model checkpoints here/'
os.makedirs(checkpoint_path_no_masks, exist_ok=True)
clip_value = 1.5  # Clipping value to prevent exploding gradients

# Early stopping parameters
early_stopping_patience = 10  # Stop after no improvement for 10 consecutive validation checks
best_validation_loss = float('inf')
early_stopping_counter = 0

# Accumulate gradients over several smaller batches to simulate a larger batch size
current_accumulation_steps = 0

# Linear decay of learning rate to zero
decay_epoch_start = epoch_to_load
total_epochs = num_epochs

# Define the lambda function for linear decay
def lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch + decay_epoch_start - total_epochs) / float(decay_epoch_start)
    return lr_l

# Initialize learning rate schedulers to linearly decay the learning rate to zero
scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
scheduler_D_H = torch.optim.lr_scheduler.LambdaLR(optimizer_D_H, lr_lambda=lambda_rule)
scheduler_D_P = torch.optim.lr_scheduler.LambdaLR(optimizer_D_P, lr_lambda=lambda_rule)

# Training function
def train_cyclegan_without_masks(
    generator_H2P, generator_P2H, discriminator_H, discriminator_P,
    train_loader_healthy_no_masks, train_loader_pathological_no_masks, val_loader_healthy_no_masks, val_loader_pathological_no_masks,
    optimizer_G, optimizer_D_H, optimizer_D_P,
    scheduler_G, scheduler_D_H, scheduler_D_P,
    criterion_identity, criterion_cycle, 
    wgan_gp_loss, clip_value, lambda_cycle, lambda_identity, 
    smooth_real_label, smooth_fake_label,
    checkpoint_path_no_masks, save_interval, sample_interval, num_epochs, early_stopping_patience,
    device
):
    num_batches = max(len(train_loader_pathological_no_masks), len(train_loader_healthy_no_masks))
