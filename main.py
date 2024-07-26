import os
from extract_patches import process_svs_files, move_processed_files
from generate_masks import process_files

# Define your paths here
svs_dir = "/your/original svs images/folder"
geojson_dir = "/your/geojson data/folder"
mask_dir = "/store binary masks here/folder"
svs_patches_dir = "/store svs images patches here/folder"
mask_patches_dir = "/store binary mask patches here/folder"

# Ensure necessary directories exist
os.makedirs(svs_patches_dir, exist_ok=True)
os.makedirs(mask_patches_dir, exist_ok=True)

# Step 1: Create TIFF binary masks
print("Creating TIFF binary masks...")
masks_info = process_files(svs_dir, geojson_dir, mask_dir)

# Step 2: Extract patches
print("Extracting patches...")
processed_svs_files, total_patches = process_svs_files(svs_dir, mask_dir, geojson_dir, svs_patches_dir, mask_patches_dir, processed_dir)

# Step 3: Move processed files
print("Moving processed files...")
moved_files_count, remaining_files_count = move_processed_files(svs_dir, processed_dir, svs_patches_dir)



