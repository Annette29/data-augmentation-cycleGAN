import os
import torch 

from generate_binary_masks import process_files
from extract_patches_with_lesions import resize_images_cv, process_svs_files as process_folder_with_lesions
from extract_patches_without_lesions import process_svs_files as process_folder_without_lesions
from train_cyclegan import initialize_components as initialize_components_masks, train_cyclegan_with_masks, visualize_activations, limit_samples, main_plotting_function as main_plotting_function_masks
from augment_original_dataset import setup_directories, generate_fake_samples_masks, generate_fake_samples_without_masks, plot_random_pairs
from preprocess_NN_training_data import load_all_datasets, combine_datasets
from classification_task import initialize_model, train_model, plot_sensitivity_vs_fp_comparison as plot_sensitivity_vs_fp_comparison_masks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
plot_dir = "/store plots for real images vs binary masks vs synthetic images here/"

# Ensure necessary directories exist
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(lesions_svs_patches_dir, exist_ok=True)
os.makedirs(without_lesions_svs_patches_dir, exist_ok=True)
os.makedirs(mask_patches_dir, exist_ok=True)
os.makedirs(resized_lesions_svs_patches_dir, exist_ok=True)
os.makedirs(resized_mask_patches_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

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

# Step 3: Extract 1024*1024 patches from class: without lesions
print("Extracting patches for images without lesions...")
total_images_processed, total_patches_extracted = process_folder_without_lesions(without_lesions_svs_dir, without_lesions_svs_patches_dir)
print(f"\nProcessed {total_images_processed} SVS images without lesions and extracted a total of {total_patches_extracted} patches.")

# Step 4: Train a CycleGAN model to synthesize pathology onto healthy images with binary masks as conditional input
(
    generator_H2P, generator_P2H, discriminator_H, discriminator_P,
    train_loader_healthy, train_loader_pathological, val_loader_healthy, val_loader_pathological, test_loader_healthy, test_loader_pathological, 
    optimizer_G, optimizer_D_H, optimizer_D_P,
    scheduler_G, scheduler_D_H, scheduler_D_P,
    criterion_identity, criterion_cycle, criterion_abnormality,
    wgan_gp_loss
) = initialize_components_masks(device)

train_cyclegan_with_masks(
    generator_H2P, generator_P2H, discriminator_H, discriminator_P,
    train_loader_healthy, train_loader_pathological, val_loader_healthy, val_loader_pathological,
    optimizer_G, optimizer_D_H, optimizer_D_P,
    scheduler_G, scheduler_D_H, scheduler_D_P,
    criterion_identity, criterion_cycle, criterion_abnormality,
    wgan_gp_loss, clip_value, lambda_cycle, lambda_identity, lambda_abnormality,
    smooth_real_label, smooth_fake_label,
    checkpoint_path, save_interval, sample_interval, num_epochs, early_stopping_patience,
    device
)

# Visualize activations to confirm if the generators are utilizing the binary masks 
visualize_activations(generator_H2P, test_loader_healthy, device)
visualize_activations(generator_P2H, test_loader_pathological, device)

# Plot random sets of images for visual inspection 
num_healthy_samples_test = len(test_loader_healthy.dataset)
test_loader_pathological = limit_samples(test_loader_pathological, num_healthy_samples_test)
main_plotting_function_masks(generator_H2P, generator_P2H, test_loader_healthy, test_loader_pathological, num_images=5, save_dir=plot_dir)

# Step 5: Evaluate the CycleGAN model using IoU and SSIM metrics 
(
    generator_H2P, test_loader_healthy, test_loader_pathological
) = initialize_components_masks(device)

avg_iou_healthy, avg_ssim_healthy = evaluate_model(generator_H2P, test_loader_healthy, device)
print(f'Average IoU (Healthy): {avg_iou_healthy}')
print(f'Average SSIM (Healthy): {avg_ssim_healthy}')

num_healthy_samples_test = len(test_loader_healthy.dataset)
test_loader_pathological = limit_samples(test_loader_pathological, num_healthy_samples_test)
avg_iou_pathological, avg_ssim_pathological = evaluate_model(generator_H2P, test_loader_pathological, device)
print(f'Average IoU (Pathological): {avg_iou_pathological}')
print(f'Average SSIM (Pathological): {avg_ssim_pathological}')

# Step 6: Train a CycleGAN model to synthesize pathology onto healthy images without any conditional input
(
    generator_H2P_no_masks, generator_P2H_no_masks, discriminator_H, discriminator_P,
    train_loader_healthy_no_masks, train_loader_pathological_no_masks, val_loader_healthy_no_masks, val_loader_pathological_no_masks,
    optimizer_G, optimizer_D_H, optimizer_D_P,
    scheduler_G, scheduler_D_H, scheduler_D_P,
    criterion_identity, criterion_cycle, 
    wgan_gp_loss
) = initialize_components_without_masks(device)

train_cyclegan_without_masks(
    generator_H2P_no_masks, generator_P2H_no_masks, discriminator_H, discriminator_P,
    train_loader_healthy_no_masks, train_loader_pathological_no_masks, val_loader_healthy_no_masks, val_loader_pathological_no_masks,
    optimizer_G, optimizer_D_H, optimizer_D_P,
    scheduler_G, scheduler_D_H, scheduler_D_P,
    criterion_identity, criterion_cycle, 
    wgan_gp_loss, clip_value, lambda_cycle, lambda_identity, 
    smooth_real_label, smooth_fake_label,
    checkpoint_path_no_masks, save_interval, sample_interval, num_epochs, early_stopping_patience,
    device    
)

# Plot random pairs of images for visual inspection 
num_healthy_samples_test_no_masks = len(test_loader_healthy_no_masks.dataset)
test_loader_pathological_no_masks = limit_samples(test_loader_pathological_no_masks, num_healthy_samples_test_no_masks)
main_plotting_function_without_masks(generator_H2P_no_masks, generator_P2H_no_masks, test_loader_healthy_no_masks, test_loader_pathological_no_masks, num_images=5, save_dir=plot_dir)

# Step 7: Add synthetic images (created from 1. a CycleGAN trained with binary masks, 2. a CycleGAN trained with no conditional input) to the original training dataset for a classification task to evaluate whether fake images improve a neural network model's generalization abilities
(
    generator_H2P, generator_P2H,
    train_image_dir_healthy, train_image_dir_pathological, train_loader_healthy, train_loader_pathological, 
    val_image_dir_healthy, val_image_dir_pathological, val_loader_healthy, val_loader_pathological, 
    test_image_dir_healthy, test_image_dir_pathological, test_loader_healthy, test_loader_pathological
) = initialize_components_masks(device)

(
    generator_H2P_no_masks, generator_P2H_no_masks,
    train_image_dir_healthy, train_image_dir_pathological, train_loader_healthy_no_masks, train_loader_pathological_no_masks, 
    val_image_dir_healthy, val_image_dir_pathological, val_loader_healthy_no_masks, val_loader_pathological_no_masks, 
    test_image_dir_healthy, test_image_dir_pathological, test_loader_healthy_no_masks, test_loader_pathological_no_masks
) = initialize_components_without_masks(device)

# Limit the number of samples in both loaders to match the loader with the fewer number of images (depends on your dataset)
num_healthy_samples_train = len(train_loader_healthy.dataset)
train_loader_pathological = limit_samples(train_loader_pathological, num_healthy_samples_train)

num_healthy_samples_val = len(val_loader_pathological.dataset)
val_loader_healthy = limit_samples(val_loader_healthy, num_healthy_samples_val)

num_healthy_samples_test = len(test_loader_healthy.dataset)
test_loader_pathological = limit_samples(test_loader_pathological, num_healthy_samples_test)

num_healthy_samples_train_no_masks = len(train_loader_healthy_no_masks.dataset)
train_loader_pathological_no_masks = limit_samples(train_loader_pathological_no_masks, num_healthy_samples_train_no_masks)

num_healthy_samples_val_no_masks = len(val_loader_pathological_no_masks.dataset)
val_loader_healthy_no_masks = limit_samples(val_loader_healthy_no_masks, num_healthy_samples_val_no_masks)

num_healthy_samples_test_no_masks = len(test_loader_healthy_no_masks.dataset)
test_loader_pathological_no_masks = limit_samples(test_loader_pathological_no_masks, num_healthy_samples_test_no_masks)

# Specify the base directory and the directory structure
base_dir = "/your/synthetic_data/folder"
main_categories = ['With Masks', 'Without Masks']
sub_categories = ['Without Lesions', 'With Lesions']
sub_dirs = ['Training Data', 'Validation Data', 'Test Data']

# Ensure necessary directories exist
setup_directories(base_dir, main_categories, sub_categories, sub_dirs)

# Directories for saving fake images
fake_A_dir = os.path.join(base_dir, 'With Masks', 'Without Lesions', 'Training Data')
fake_B_dir = os.path.join(base_dir, 'With Masks', 'With Lesions', 'Training Data')
fake_C_dir = os.path.join(base_dir, 'With Masks', 'Without Lesions', 'Validation Data')
fake_D_dir = os.path.join(base_dir, 'With Masks', 'With Lesions', 'Validation Data')
fake_E_dir = os.path.join(base_dir, 'With Masks', 'Without Lesions', 'Test Data')
fake_F_dir = os.path.join(base_dir, 'With Masks', 'With Lesions', 'Test Data')
fake_G_dir = os.path.join(base_dir, 'Without Masks', 'Without Lesions', 'Training Data')
fake_H_dir = os.path.join(base_dir, 'Without Masks', 'With Lesions', 'Training Data')
fake_I_dir = os.path.join(base_dir, 'Without Masks', 'Without Lesions', 'Validation Data')
fake_J_dir = os.path.join(base_dir, 'Without Masks', 'With Lesions', 'Validation Data')
fake_K_dir = os.path.join(base_dir, 'Without Masks', 'Without Lesions', 'Test Data')
fake_L_dir = os.path.join(base_dir, 'Without Masks', 'With Lesions', 'Test Data')

# Generate fake samples for the respective data loaders and directories & select random images and their corresponding fakes from each dataset, and then plot them
# For With Masks
generate_fake_samples_masks(train_loader_healthy, generator_H2P, fake_A_dir, device)
plot_random_pairs(train_image_dir_healthy, fake_A_dir)
generate_fake_samples_masks(train_loader_pathological, generator_P2H, fake_B_dir, device)
plot_random_pairs(train_image_dir_pathological, fake_B_dir)
generate_fake_samples_masks(val_loader_healthy, generator_H2P, fake_C_dir, device)
plot_random_pairs(val_image_dir_healthy, fake_C_dir)
generate_fake_samples_masks(val_loader_pathological, generator_P2H, fake_D_dir, device)
plot_random_pairs(val_image_dir_pathological, fake_D_dir)
generate_fake_samples_masks(test_loader_healthy, generator_H2P, fake_E_dir, device)
plot_random_pairs(test_image_dir_healthy, fake_E_dir)
generate_fake_samples_masks(test_loader_pathological, generator_P2H, fake_F_dir, device)
plot_random_pairs(test_image_dir_pathological, fake_F_dir)

# For Without Masks
generate_fake_samples_without_masks(train_loader_healthy_no_masks, generator_H2P_no_masks, fake_G_dir, device)
plot_random_pairs(train_image_dir_healthy, fake_G_dir)
generate_fake_samples_without_masks(train_loader_pathological_no_masks, generator_P2H_no_masks, fake_H_dir, device)
plot_random_pairs(train_image_dir_pathological, fake_H_dir)
generate_fake_samples_without_masks(val_loader_healthy_no_masks, generator_H2P_no_masks, fake_I_dir, device)
plot_random_pairs(val_image_dir_healthy, fake_I_dir)
generate_fake_samples_without_masks(val_loader_pathological_no_masks, generator_P2H_no_masks, fake_J_dir, device)
plot_random_pairs(val_image_dir_pathological, fake_J_dir)
generate_fake_samples_without_masks(test_loader_healthy_no_masks, generator_H2P_no_masks, fake_K_dir, device)
plot_random_pairs(test_image_dir_healthy, fake_K_dir)
generate_fake_samples_without_masks(test_loader_pathological_no_masks, generator_P2H_no_masks, fake_L_dir, device)
plot_random_pairs(test_image_dir_pathological, fake_L_dir)

# Step 8: Train 4 independent sets of models and measure the sensitivity of models trained with real data only, synthetic data only, real + synthetic data for fake images created from a CycleGAN trained with binary masks, and real + synthetic data for fake images created from a CycleGAN trained without binary masks
base_dir = '/the OG folder with the entire BRACS Dataset (real and synthetic)/'
datasets, counts = load_all_datasets(base_dir)
train_combined, val_combined, test_combined = combine_datasets(datasets, counts)

train_combined_with_masks = combined_datasets['train_combined_with_masks']
val_combined_with_masks = combined_datasets['val_combined_with_masks']
test_combined_with_masks = combined_datasets['test_combined_with_masks']

train_combined_without_masks = combined_datasets['train_combined_without_masks']
val_combined_without_masks = combined_datasets['val_combined_without_masks']
test_combined_without_masks = combined_datasets['test_combined_without_masks']

# Initialize model, optimizer, scheduler
weights = None  # Set to None or a path to weights
model, criterion, optimizer, scheduler = initialize_model(weights)

# Train the model
best_model_real, sensitivity_progression_real, false_positives_progression_real = train_model(
    model,
    train_real, val_real,
    criterion, optimizer, scheduler,
    num_epochs= 100, # Start by training for 100 epochs and observe the resulting output
    batch_size=32,
    threshold=0.5
)

best_model_synthetic, sensitivity_progression_synthetic, false_positives_progression_synthetic = train_model(
    model,
    train_synthetic, val_synthetic,
    criterion, optimizer, scheduler,
    num_epochs= 100, # Start by training for 100 epochs and observe the resulting output
    batch_size=32,
    threshold=0.5
)

best_model_combined_masks, sensitivity_progression_combined_masks, false_positives_progression_combined_masks = train_model(
    model,
    train_combined_masks, val_combined_masks,
    criterion, optimizer, scheduler,
    num_epochs= 100, # Start by training for 100 epochs and observe the resulting output
    batch_size=32,
    threshold=0.5
)

best_model_combined_without_masks, sensitivity_progression_combined_without_masks, false_positives_progression_combined_without_masks = train_model(
    model,
    train_combined_without_masks, val_combined_without_masks,
    criterion, optimizer, scheduler,
    num_epochs= 100, # Start by training for 100 epochs and observe the resulting output
    batch_size=32,
    threshold=0.5
)

# Plot the sensitivity vs false positives comparison (using real, synthetic, and combined_masks data)
plot_sensitivity_vs_fp_comparison_masks(
    sensitivity_progression_real, false_positives_progression_real,
    sensitivity_progression_synthetic, false_positives_progression_synthetic,
    sensitivity_progression_combined_masks, false_positives_progression_combined_masks
)

# Plot the sensitivity vs false positives comparison (using real, synthetic, and combined_without_masks data)
plot_sensitivity_vs_fp_comparison_without_masks(
    sensitivity_progression_real, false_positives_progression_real,
    sensitivity_progression_synthetic, false_positives_progression_synthetic,
    sensitivity_progression_combined_without_masks, false_positives_progression_combined_without_masks
)
