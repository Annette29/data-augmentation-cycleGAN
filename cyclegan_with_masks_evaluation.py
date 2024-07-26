from model_with_masks_architectures import UNetResNet34
from train_cyclegan_with_masks import initialize_components

import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
from torchvision import transforms
from PIL import Image

(
    generator_H2P, test_loader_healthy, test_loader_pathological
) = initialize_components(device)

# Load previously-saved model checkpoint
checkpoint_path = '/your/model checkpoints/folder'
epoch_to_load = #latest_model_checkpoint_saved
generator_H2P.load_state_dict(torch.load(os.path.join(checkpoint_path, f'generator_H2P_epoch{epoch_to_load}.pth')))
generator_H2P.eval()

# Compute SSIM (Structural Similarity Index Measure)
def compute_ssim(image1, image2):
    image1 = (image1 * 255).astype(np.uint8)
    image2 = (image2 * 255).astype(np.uint8)
    ssim_value = ssim(image1, image2, win_size=7, channel_axis=-1)
    return ssim_value

# Compute IoU (Intersection over Union)
def compute_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask)
    union = np.logical_or(pred_mask, true_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# Convert image to binary mask
def convert_to_binary_mask(image, threshold=0.3):
    binary_mask = (image > threshold).astype(np.uint8)
    return binary_mask

# Evaluate model
def evaluate_model(generator, test_loader, device):
    iou_scores = []
    ssim_scores = []

    for real_img, mask, image_name in test_loader:
        with torch.no_grad():
            real_img = real_img.to(device)
            mask = mask.to(device)

            # Generate fake pathological image
            fake_pathology = generator(real_img, mask)

        for i in range(real_img.size(0)):
            real_img_np = real_img[i].cpu().numpy().squeeze()  # (C, H, W)
            mask_np = mask[i].cpu().numpy().squeeze()          # (H, W)
            fake_pathology_np = fake_pathology[i].cpu().numpy().squeeze()  # (C, H, W)

            # Convert to grayscale if RGB
            if fake_pathology_np.shape[0] == 3:
                fake_pathology_gray = np.mean(fake_pathology_np, axis=0)
            else:
                fake_pathology_gray = fake_pathology_np[0]  # Single-channel

            # Convert to (H, W, C) if necessary
            if real_img_np.ndim == 3:
                real_img_np = real_img_np.transpose(1, 2, 0)
            if fake_pathology_np.ndim == 3:
                fake_pathology_np = fake_pathology_np.transpose(1, 2, 0)

            # Convert to binary mask
            fake_pathology_mask = convert_to_binary_mask(fake_pathology_gray)

            # Ensure the masks are binary
            mask_np = (mask_np > 0.5).astype(np.uint8)

            # Compute IoU and SSIM
            iou = compute_iou(fake_pathology_mask, mask_np)
            iou_scores.append(iou)

            ssim_value = compute_ssim(real_img_np, fake_pathology_np)
            ssim_scores.append(ssim_value)

    avg_iou = np.mean(iou_scores)
    avg_ssim = np.mean(ssim_scores)

    return avg_iou, avg_ssim

