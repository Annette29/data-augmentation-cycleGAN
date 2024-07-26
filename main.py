import os
from generate_binary_masks import process_files
from extract_patches_with_lesions import process_svs_files as process_folder_with_lesions, resize_images_cv
from extract_patches_without_lesions import process_svs_files as process_folder_without_lesions
from load_training_dataset import
from model_architectures import
from train_cyclegan_with_masks import 
from cyclegan_with_masks_evaluation import 
from train_cyclegan_without_masks import 
from augment_original_dataset import 
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

# Ensure necessary directories exist
os.makedirs(lesions_svs_patches_dir, exist_ok=True)
os.makedirs(mask_patches_dir, exist_ok=True)
os.makedirs(resized_lesions_svs_patches_dir, exist_ok=True)
os.makedirs(resized_mask_patches_dir, exist_ok=True)

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

# Step 5: Evaluate the CycleGAN model using IoU and SSIM metrics 

# Step 6: Train a CycleGAN model to synthesize pathology onto healthy images without any conditional input

# Step 7: Add synthetic images (created from a CycleGAN trained with binary masks) to the original training dataset for a classification task to evaluate whether fake images improve a neural network model's generalization abilities 

# Step 8: Add synthetic images (created from a CycleGAN trained without binary masks) to the original training dataset for a classification task to evaluate whether fake images improve a neural network model's generalization abilities 

# Step 9: Train 3 independent sets of models and measure the sensitivity of models trained with real data only, synthetic data only, and real + synthetic data for fake images created from a CycleGAN trained with binary masks

# Step 10: Train 3 independent sets of models and measure the sensitivity of models trained with real data only, synthetic data only, and real + synthetic data for fake images created from a CycleGAN trained without binary masks



