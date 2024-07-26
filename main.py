import os
import torch 
from generate_binary_masks import process_files
from extract_patches_with_lesions import process_svs_files as process_folder_with_lesions, resize_images_cv
from extract_patches_without_lesions import process_svs_files as process_folder_without_lesions
from preprocess_training_data import create_dataloaders
from model_architectures import UNetResNet34, PatchGANDiscriminator, weights_init_normal, WassersteinLossGP, CombinedL1L2Loss, AbnormalityMaskLoss
from train_cyclegan_with_masks import 
from cyclegan_with_masks_evaluation import 
from train_cyclegan_without_masks import 
from augment_original_dataset_with_masks import 
from augment_original_dataset_without_masks import 
from classification_task_with_masks import 
from classification_task_without_masks import 

# Define your paths here
lesions_svs_dir = "/your/original svs images with lesions/folder"
without_lesions_svs_dir = "/your/original svs images without lesions/folder"
geojson_dir = "/your/geojson data/folder"
mask_dir = "/store binary masks for svs images with lesions here/"
lesions_svs_patches_dir = "/store patches for svs images with lesions here/"
without_lesions_svs_patches_dir = "/store patches for svs images without lesions here/"
mask_patches_dir = "/store binary mask patches here/"
resized_lesions_svs_patches_dir = "/store 1024*1024 patches for svs images with lesions here/"
resized_mask_patches_dir = "/store 1024*1024 binary mask patches here/"
checkpoint_path = '/store/model checkpoints here/'

# Ensure necessary directories exist
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(lesions_svs_patches_dir, exist_ok=True)
os.makedirs(without_lesions_svs_patches_dir, exist_ok=True)
os.makedirs(mask_patches_dir, exist_ok=True)
os.makedirs(resized_lesions_svs_patches_dir, exist_ok=True)
os.makedirs(resized_mask_patches_dir, exist_ok=True)
os.makedirs(checkpoint_path, exist_ok=True)

# Step 1: Create TIFF binary masks for images with lesions
print("Creating TIFF binary masks for images with lesions...")
masks_info = process_files(lesions_svs_dir, geojson_dir, mask_dir)

# Step 2: Extract patches from class: with lesions
print("Extracting patches from images with lesions...")
processed_svs_files, total_patches = process_folder_with_lesions(lesions_svs_dir, mask_dir, geojson_dir, lesions_svs_patches_dir, mask_patches_dir, processed_dir)
print(f"\nProcessed {processed_svs_files} SVS images with lesions and extracted a total of {total_patches} patches.")

processed_count, skipped_count = resize_images_cv(lesions_svs_patches_dir, resized_lesions_svs_patches_dir)
print(f"Number of svs patches with lesions processed: {processed_count}")
print(f"Number of svs patches with lesions skipped due to errors: {skipped_count}")

processed_count, skipped_count = resize_images_cv(mask_patches_dir, resized_mask_patches_dir)
print(f"Number of binary mask patches processed: {processed_count}")
print(f"Number of binary mask patches skipped due to errors: {skipped_count}")

# Step 3: Extract patches from class: without lesions
print("Extracting patches for images without lesions...")
total_images_processed, total_patches_extracted = process_folder_without_lesions(without_lesions_svs_dir, without_lesions_svs_patches_dir)
print(f"\nProcessed {total_images_processed} SVS images without lesions and extracted a total of {total_patches_extracted} patches.")

# Step 4: Train a CycleGAN model to synthesize pathology onto healthy images with binary masks as conditional input
# Step 4.1: Load the training data 
# Define root directories
base_dir = 'your/patches for svs images both with and without lesions/folder'
mask_base_dir = '/your/binary mask patches/folder'

# Without Lesions
train_image_dir_healthy = os.path.join(base_dir, 'Without Lesions/Training Data')
train_mask_dir_healthy = os.path.join(mask_base_dir, 'Resized With Lesions/Training Data')
healthy_mask_filenames_train = [f for f in os.listdir(train_mask_dir_healthy) if f.endswith('.png') or f.endswith('.jpg')]
train_loader_healthy = create_dataloaders(train_image_dir_healthy, train_mask_dir_healthy,
                                           mask_name_func=lambda image_name: random_pathological_mask_name_func(image_name, healthy_mask_filenames_train))


val_image_dir_healthy = os.path.join(base_dir, 'Without Lesions/Validation Data')
val_mask_dir_healthy = os.path.join(mask_base_dir, 'Resized With Lesions/Validation Data')
healthy_mask_filenames_val = [f for f in os.listdir(val_mask_dir_healthy) if f.endswith('.png') or f.endswith('.jpg')]
val_loader_healthy = create_dataloaders(val_image_dir_healthy, val_mask_dir_healthy,
                                        mask_name_func=lambda image_name: random_pathological_mask_name_func(image_name, healthy_mask_filenames_val),
                                        shuffle=False, random_sampling=True)

test_image_dir_healthy = os.path.join(base_dir, 'Without Lesions/Test Data')
test_mask_dir_healthy = os.path.join(mask_base_dir, 'Resized With Lesions/Test Data')
healthy_mask_filenames_test = [f for f in os.listdir(test_mask_dir_healthy) if f.endswith('.png') or f.endswith('.jpg')]
test_loader_healthy = create_dataloaders(test_image_dir_healthy, test_mask_dir_healthy,
                                         mask_name_func=lambda image_name: random_pathological_mask_name_func(image_name, healthy_mask_filenames_test),
                                         shuffle=False, random_sampling=True)

# With Lesions
train_loader_pathological = create_dataloaders(os.path.join(base_dir, 'Resized With Lesions/Training Data'),
                                               os.path.join(mask_base_dir, 'Resized With Lesions/Training Data'),
                                               mask_name_func=default_mask_name_func)
val_loader_pathological = create_dataloaders(os.path.join(base_dir, 'Resized With Lesions/Validation Data'),
                                             os.path.join(mask_base_dir, 'Resized With Lesions/Validation Data'),
                                             mask_name_func=default_mask_name_func, shuffle=False, random_sampling=True)
test_loader_pathological = create_dataloaders(os.path.join(base_dir, 'Resized With Lesions/Test Data'),
                                              os.path.join(mask_base_dir, 'Resized With Lesions/Test Data'),
                                              mask_name_func=default_mask_name_func, shuffle=False, random_sampling=True)

# Step 4.2: Initialize the 4 models for use - generator_H2P, generator_p2H, discriminator_H, and disciminator_P & the 4 loss functions to train them - WGAN-GP as the adversarial loss. identity loss, cycle consistency loss, and abnormality loss
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

# Step 4.3: 

# Step 5: Evaluate the CycleGAN model using IoU and SSIM metrics 

# Step 6: Train a CycleGAN model to synthesize pathology onto healthy images without any conditional input

# Step 7: Add synthetic images (created from a CycleGAN trained with binary masks) to the original training dataset for a classification task to evaluate whether fake images improve a neural network model's generalization abilities 

# Step 8: Add synthetic images (created from a CycleGAN trained without binary masks) to the original training dataset for a classification task to evaluate whether fake images improve a neural network model's generalization abilities 

# Step 9: Train 3 independent sets of models and measure the sensitivity of models trained with real data only, synthetic data only, and real + synthetic data for fake images created from a CycleGAN trained with binary masks

# Step 10: Train 3 independent sets of models and measure the sensitivity of models trained with real data only, synthetic data only, and real + synthetic data for fake images created from a CycleGAN trained without binary masks



