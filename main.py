import os
from generate_masks import process_files
from extract_patches import process_svs_files, resize_images_cv

# Define your paths here
svs_dir = "/your/original svs images/folder"
geojson_dir = "/your/geojson data/folder"
mask_dir = "/store binary masks here/"
svs_patches_dir = "/store svs images patches here/"
mask_patches_dir = "/store binary mask patches here/"
resized_svs_patches_dir = "/store 1024*1024 svs patches here/"
resized_mask_patches_dir = "/store 1024*1024 binary mask patches here/"

# Ensure necessary directories exist
os.makedirs(svs_patches_dir, exist_ok=True)
os.makedirs(mask_patches_dir, exist_ok=True)
os.makedirs(resized_svs_patches_dir, exist_ok=True)
os.makedirs(resized_mask_patches_dir, exist_ok=True)

# Step 1: Create TIFF binary masks
print("Creating TIFF binary masks for images with lesions...")
masks_info = process_files(svs_dir, geojson_dir, mask_dir)

# Step 2: Extract patches from class: with lesions
print("Extracting patches from images with lesions...")
processed_svs_files, total_patches = process_svs_files(svs_dir, mask_dir, geojson_dir, svs_patches_dir, mask_patches_dir, processed_dir)
print(f"\nProcessed {processed_svs_files} SVS files and extracted a total of {total_patches} patches.")
resize_images_cv(svs_patches_dir, resized_svs_patches_dir)
resize_images_cv(mask_patches_dir, resized_mask_patches_dir)

# Step 3: Extract patches from class: without lesions




