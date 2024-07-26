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

device = "cuda" if torch.cuda.is_available() else "cpu"

from model_architectures import UNetResNet34, PatchGANDiscriminator, weights_init_normal, WassersteinLossGP, CombinedL1L2Loss, AbnormalityMaskLoss

# Initialize the 4 models for use - generator_H2P, generator_p2H, discriminator_H, and disciminator_P & the 4 loss functions to train them - WGAN-GP as the adversarial loss. identity loss, cycle consistency loss, and abnormality loss
generator_H2P = UNetResNet34(in_channels=4, out_channels=3).to(device)  # Perform element-wise multiplication of RGB images with binary masks selected randomly, hence in_channels=4
generator_P2H = UNetResNet34(in_channels=4, out_channels=3).to(device)  # Perform element-wise multiplication of RGB images with their corresponding binary masks, hence in_channels=4
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
criterion_abnormality = AbnormalityMaskLoss(lambda_l1=1.0, lambda_l2=1.0).to(device)

# Initialize optimizers with reducing learning rate and weight decay/L2 regularization
optimizer_G = torch.optim.Adam(itertools.chain(generator_H2P.parameters(), generator_P2H.parameters()), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.001)
optimizer_D_H = torch.optim.Adam(discriminator_H.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.001)
optimizer_D_P = torch.optim.Adam(discriminator_P.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.001)

# Wrap models with DataParallel for data parallelism (allows you to split the batch of data across multiple GPUs and compute the results in parallel)
generator_H2P = torch.nn.DataParallel(generator_H2P)
generator_P2H = torch.nn.DataParallel(generator_P2H)
discriminator_H = torch.nn.DataParallel(discriminator_H)
discriminator_P = torch.nn.DataParallel(discriminator_P)

# Define the validaton function
def validate(generator_H2P, generator_P2H, val_loader_healthy, val_loader_pathological, criterion_cycle, criterion_identity, criterion_abnormality):
    generator_H2P.eval()
    generator_P2H.eval()

    total_loss_id_A = 0.0
    total_loss_id_B = 0.0
    total_loss_cycle_ABA = 0.0
    total_loss_cycle_BAB = 0.0
    total_loss_abnormality = 0.0
    total_samples = 0

    # Initialize iterators
    pathological_iter = iter(val_loader_pathological)
    healthy_iter = iter(val_loader_healthy)

    with torch.no_grad():
        for _ in range(max(len(val_loader_pathological), len(val_loader_healthy))):
            try:
                # Fetch pathological data
                real_B, mask_B, image_name_B = next(pathological_iter)
            except StopIteration:
                # Reset iterator if StopIteration occurs
                pathological_iter = iter(val_loader_pathological)
                real_B, mask_B, image_name_B = next(pathological_iter)

            real_B = real_B.to(device)
            mask_B = mask_B.to(device)

            try:
                # Fetch healthy data
                real_A, mask_A, image_name_A = next(healthy_iter)
            except StopIteration:
                # Reset iterator if StopIteration occurs
                healthy_iter = iter(val_loader_healthy)
                real_A, mask_A, image_name_A = next(healthy_iter)

            real_A = real_A.to(device)
            mask_A = mask_A.to(device)

            # Ensure consistent batch sizes
            if real_A.size(0) != real_B.size(0):
                continue

            # 1. Identity Loss
            empty_mask = torch.zeros_like(mask_A)
            loss_id_A = criterion_identity(generator_P2H(real_A, empty_mask), real_A)
            loss_id_B = criterion_identity(generator_H2P(real_B, empty_mask), real_B)

            # 2. Cycle Consistency Loss
            fake_B = generator_H2P(real_A.detach(), mask_A.detach()).clone()
            recov_A = generator_P2H(fake_B, mask_A).clone()
            loss_cycle_ABA = criterion_cycle(recov_A, real_A)

            fake_A = generator_P2H(real_B.detach(), mask_B.detach()).clone()
            recov_B = generator_H2P(fake_A, mask_B).clone()
            loss_cycle_BAB = criterion_cycle(recov_B, real_B.detach())

            # 3. Abnormality Mask Loss (for PHP cycle only)
            loss_abnormality = criterion_abnormality(fake_A, real_B, mask_B)

            # Accumulate total losses
            batch_size_A = real_A.size(0)
            batch_size_B = real_B.size(0)
            total_loss_id_A += loss_id_A.item() * batch_size_A
            total_loss_id_B += loss_id_B.item() * batch_size_B
            total_loss_cycle_ABA += loss_cycle_ABA.item() * batch_size_A
            total_loss_cycle_BAB += loss_cycle_BAB.item() * batch_size_B
            total_loss_abnormality += loss_abnormality.item() * batch_size_A
            total_samples += batch_size_A + batch_size_B  # Counting batches processed

    # Calculate average losses
    avg_loss_id_A = total_loss_id_A / total_samples
    avg_loss_id_B = total_loss_id_B / total_samples
    avg_loss_cycle_ABA = total_loss_cycle_ABA / total_samples
    avg_loss_cycle_BAB = total_loss_cycle_BAB / total_samples
    avg_loss_abnormality = total_loss_abnormality / total_samples

    generator_H2P.train()
    generator_P2H.train()

    return avg_loss_id_A, avg_loss_id_B, avg_loss_cycle_ABA, avg_loss_cycle_BAB, avg_loss_abnormality
    


# Define constants and hyperparameters
epoch_counter = 0
num_epochs = 1001 # Start by training for 1000 epochs and observe the resulting outputs 
lambda_cycle = 10.0
lambda_identity = 10.0
lambda_abnormality = 10.0
sample_interval = 250  # Adjusted for less frequent loss calculation
save_interval = 200  # Save model states every 200 epochs
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
def train_cyclegan_with_masks(
    generator_H2P, generator_P2H, discriminator_H, discriminator_P,
    train_loader_healthy, train_loader_pathological, val_loader_healthy, val_loader_pathological,
    optimizer_G, optimizer_D_H, optimizer_D_P,
    scheduler_G, scheduler_D_H, scheduler_D_P,
    criterion_identity, criterion_cycle, criterion_abnormality,
    wgan_gp_loss, clip_value, lambda_cycle, lambda_identity, lambda_abnormality,
    smooth_real_label, smooth_fake_label,
    checkpoint_path, save_interval, sample_interval, num_epochs, early_stopping_patience,
    device
):
    num_batches = max(len(train_loader_pathological), len(train_loader_healthy))
    
    while epoch_counter < num_epochs and early_stopping_counter < early_stopping_patience:
        # Create iterators for the dataloaders
        pathological_iter = iter(train_loader_pathological)
        healthy_iter = iter(train_loader_healthy)
    
        for _ in range(num_batches):
            try:
                # Fetch the next batch of pathological images
                real_B, mask_B, image_name_B = next(pathological_iter)
            except StopIteration:
                pathological_iter = iter(train_loader_pathological)
                real_B, mask_B, image_name_B = next(pathological_iter)
    
            real_B = real_B.to(device)
            mask_B = mask_B.to(device)
    
            try:
                # Fetch the next batch of healthy images
                real_A, mask_A, image_name_A = next(healthy_iter)
            except StopIteration:
                healthy_iter = iter(train_loader_healthy)
                real_A, mask_A, image_name_A = next(healthy_iter)
    
            real_A = real_A.to(device)
            mask_A = mask_A.to(device)
    
            # Ensure consistent batch sizes
            if real_A.size(0) != real_B.size(0):
                continue
    
            # Train Generators and Discriminators
            optimizer_G.zero_grad()
            optimizer_D_H.zero_grad()
            optimizer_D_P.zero_grad()
    
            # 1. Identity Loss
            empty_mask = torch.zeros_like(mask_A)
            loss_id_A = criterion_identity(generator_P2H(real_A, empty_mask), real_A)
            loss_id_B = criterion_identity(generator_H2P(real_B, empty_mask), real_B)
    
            # 2. Main adversarial loss
            fake_B = generator_H2P(real_A.detach(), mask_A.detach()).clone()
            fake_A = generator_P2H(real_B.detach(), mask_B.detach()).clone()
    
            loss_GAN_A2B = wgan_gp_loss(discriminator_P, real_B, fake_B)
            loss_GAN_B2A = wgan_gp_loss(discriminator_H, real_A, fake_A)
    
            # 3. Cycle Consistency Loss
            recov_A = generator_P2H(fake_B, mask_A).clone()
            loss_cycle_ABA = criterion_cycle(recov_A, real_A.detach())
    
            recov_B = generator_H2P(fake_A, mask_B).clone()
            loss_cycle_BAB = criterion_cycle(recov_B, real_B.detach())
    
            # 4. Abnormality Mask Loss (for PHP cycle only)
            loss_abnormality = criterion_abnormality(fake_A, real_B, mask_B)
    
            # Total generators' loss
            loss_G = (loss_GAN_A2B + loss_GAN_B2A) + \
                     lambda_cycle * (loss_cycle_ABA + loss_cycle_BAB) + \
                     lambda_identity * (loss_id_A + loss_id_B) + \
                     lambda_abnormality * loss_abnormality
    
            loss_G.backward(retain_graph=True)
    
            if clip_value > 0:
                clip_grad_norm_(generator_H2P.parameters(), clip_value)
                clip_grad_norm_(generator_P2H.parameters(), clip_value)
    
            # Update generator every 16 steps
            if current_accumulation_steps % 16 == 0:
                optimizer_G.step()
                optimizer_G.zero_grad()
    
            # Main adversarial loss to train both discriminators
            loss_D_H = wgan_gp_loss(discriminator_H, real_A, fake_A.detach().clone(), smooth_real_label, smooth_fake_label, apply_label_smoothing=True)
            loss_D_P = wgan_gp_loss(discriminator_P, real_B, fake_B.detach().clone(), smooth_real_label, smooth_fake_label, apply_label_smoothing=True)
    
            # Backward pass and update for discriminators
            loss_D_H.backward()
            loss_D_P.backward()
    
            current_accumulation_steps += 1
    
            # Update discriminator every 4 steps
            if current_accumulation_steps % 4 == 0:
                optimizer_D_H.step()
                optimizer_D_P.step()
                optimizer_D_H.zero_grad()
                optimizer_D_P.zero_grad()
    
                if clip_value > 0:
                    clip_grad_norm_(discriminator_H.parameters(), clip_value)
                    clip_grad_norm_(discriminator_P.parameters(), clip_value)
    
        # End of epoch
        epoch_counter += 1     # Update epoch counter after processing all batches for the current epoch
    
        # Update learning rates according to the linear decay schedule
        scheduler_G.step()
        scheduler_D_H.step()
        scheduler_D_P.step()
    
        # Save model checkpoints
        if epoch_counter % save_interval == 0:
            # Save checkpoints
            torch.save(generator_H2P.module.state_dict(), os.path.join(checkpoint_path, f'generator_H2P_epoch{epoch_counter}.pth'))
            torch.save(generator_P2H.module.state_dict(), os.path.join(checkpoint_path, f'generator_P2H_epoch{epoch_counter}.pth'))
            torch.save(discriminator_H.module.state_dict(), os.path.join(checkpoint_path, f'discriminator_H_epoch{epoch_counter}.pth'))
            torch.save(discriminator_P.module.state_dict(), os.path.join(checkpoint_path, f'discriminator_P_epoch{epoch_counter}.pth'))
    
            torch.save(optimizer_G.state_dict(), os.path.join(checkpoint_path, f'optimizer_G_epoch{epoch_counter}.pth'))
            torch.save(optimizer_D_H.state_dict(), os.path.join(checkpoint_path, f'optimizer_D_H_epoch{epoch_counter}.pth'))
            torch.save(optimizer_D_P.state_dict(), os.path.join(checkpoint_path, f'optimizer_D_P_epoch{epoch_counter}.pth'))
    
        # Calculate G & D loss + Validation loss
        if epoch_counter % sample_interval == 0:
            # Display discriminator and generator losses
            print(f"[Epoch {epoch_counter}/{num_epochs - 1}] "
                  f"[D loss: {loss_D_H.item() + loss_D_P.item()}] "
                  f"[G loss: {loss_G.item()}]")
    
            # Evaluate on validation set
            avg_loss_id_A, avg_loss_id_B, avg_loss_cycle_ABA, avg_loss_cycle_BAB, avg_loss_abnormality = \
                validate(generator_H2P, generator_P2H, val_loader_healthy, val_loader_pathological,
                        criterion_cycle, criterion_identity, criterion_abnormality)
    
            prev_validation_loss = avg_loss_id_A + avg_loss_id_B + avg_loss_cycle_ABA + avg_loss_cycle_BAB + avg_loss_abnormality
            print(f"[Epoch {epoch_counter}/{num_epochs - 1}] Validation Loss: {prev_validation_loss}")
            print('-' * 10)
    
            # Early stopping check + update best validation loss and early stopping counter
            if prev_validation_loss < best_validation_loss:
                best_validation_loss = prev_validation_loss
                early_stopping_counter = 0  # Reset counter
            else:
                early_stopping_counter += 1
    
        # Clear memory after each epoch
        del real_A, mask_A, fake_A, fake_B, recov_A, recov_B
        torch.cuda.empty_cache()
        gc.collect()
    
    if early_stopping_counter >= early_stopping_patience:
        print(f"Training stopped early due to no improvement in validation loss after {early_stopping_patience} epochs.")
    else:
        print("Training finished successfully.")
