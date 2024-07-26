import os
import numpy as np
import cv2
import random
import openslide

# Function to check if the patch is a background
def is_background(patch, intensity_threshold=85, background_threshold=97.5):
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    high_intensity_pixels = np.sum(gray_patch > (intensity_threshold * 255 / 100))
    total_pixels = patch.shape[0] * patch.shape[1]
    high_intensity_percentage = (high_intensity_pixels / total_pixels) * 100
    return high_intensity_percentage > background_threshold

# Function to check focus quality using the variance of the Laplacian
def is_in_focus(patch, focus_threshold=200.0):
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray_patch, cv2.CV_64F).var()
    return laplacian_var >= focus_threshold

# Function to extract and save patches
def extract_patches(slide_path, output_dir, patch_size=1024, step_size=1024, intensity_threshold=85, background_threshold=97.5, focus_threshold=200.0, sample_fraction=0.01, min_patches=20, min_extracted_patches=10):
    slide = openslide.OpenSlide(slide_path)
    width, height = slide.dimensions

    basename = os.path.splitext(os.path.basename(slide_path))[0]
    patch_output_dir = os.path.join(output_dir, basename)

    os.makedirs(patch_output_dir, exist_ok=True)

    patches = [(x, y) for y in range(0, height, step_size) for x in range(0, width, step_size)]

    if len(patches) > min_patches:
        num_patches_to_sample = max(min_extracted_patches, int(len(patches) * sample_fraction))
        patches = random.sample(patches, num_patches_to_sample)

    patch_count = 0

    for x, y in patches:
        patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert('RGB')
        patch_np = np.array(patch)

        if not is_background(patch_np, intensity_threshold, background_threshold) and is_in_focus(patch_np, focus_threshold):
            patch_filename = f"{basename}_x={x}_y={y}.png"
            patch_filepath = os.path.join(patch_output_dir, patch_filename)
            cv2.imwrite(patch_filepath, cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR))
            patch_count += 1

    return basename, patch_count

# Function to process all SVS files in a folder
def process_folder(folder_path, output_dir):
    total_images_processed = 0
    total_patches_extracted = 0

    svs_files = [f for f in os.listdir(folder_path) if f.endswith(".svs")]
    total_svs_files = len(svs_files)

    for idx, filename in enumerate(svs_files):
        slide_path = os.path.join(folder_path, filename)
        basename, patches_extracted = extract_patches(slide_path, output_dir)
        total_images_processed += 1
        total_patches_extracted += patches_extracted

        print(f"Processed SVS file {idx + 1}/{total_svs_files}: {basename}. Extracted {patches_extracted} patches.")

    print(f"Total images processed: {total_images_processed}")
    print(f"Total patches extracted: {total_patches_extracted}")

