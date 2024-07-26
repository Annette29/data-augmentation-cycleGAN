import os
import json
import numpy as np
from PIL import Image
import openslide
import shutil

Image.MAX_IMAGE_PIXELS = None

def extract_patches(slide, mask, geojson_data, base_name, svs_patches_dir, mask_patches_dir, margin=400):
    patches = []

    for feature in geojson_data:
        coords = feature['geometry']['coordinates'][0]
        coords = np.array(coords, dtype=np.int32)

        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)

        x_min = max(x_min - margin, 0)
        y_min = max(y_min - margin, 0)
        x_max = min(x_max + margin, slide.dimensions[0])
        y_max = min(y_max + margin, slide.dimensions[1])

        svs_patch = slide.read_region((x_min, y_min), 0, (x_max - x_min, y_max - y_min))
        svs_patch = svs_patch.convert("RGB")

        mask_patch = mask[y_min:y_max, x_min:x_max]

        svs_patch_path = os.path.join(svs_patches_dir, f'{base_name}_x={x_min}_y={y_min}.png')
        mask_patch_path = os.path.join(mask_patches_dir, f'{base_name}_x={x_min}_y={y_min}_mask.png')

        svs_patch.save(svs_patch_path)
        mask_patch_img = Image.fromarray(mask_patch)
        mask_patch_img.save(mask_patch_path)

        patches.append((svs_patch_path, mask_patch_path))

    return patches

def process_svs_files(svs_dir, mask_dir, geojson_dir, svs_patches_dir, mask_patches_dir, processed_dir):
    total_svs_files = len([f for f in os.listdir(svs_dir) if f.endswith('.svs')])
    processed_svs_files = 0
    total_patches = 0

    for svs_file in os.listdir(svs_dir):
        if svs_file.endswith('.svs'):
            base_name = os.path.splitext(svs_file)[0]
            slide = openslide.OpenSlide(os.path.join(svs_dir, svs_file))

            mask_path = os.path.join(mask_dir, f'{base_name}_mask.tif')
            if not os.path.exists(mask_path):
                print(f"Mask file not found for {base_name}. Skipping...")
                continue

            try:
                mask = Image.open(mask_path)
                mask = np.array(mask)
            except OSError as e:
                print(f"Error loading {mask_path}: {e}")
                continue

            geojson_path = os.path.join(geojson_dir, f'{base_name}.geojson')
            if not os.path.exists(geojson_path):
                print(f"GeoJSON file not found for {base_name}. Skipping...")
                continue

            with open(geojson_path) as f:
                geojson_data = json.load(f)

            if not isinstance(geojson_data, list):
                geojson_data = [geojson_data]

            patches = extract_patches(slide, mask, geojson_data, base_name, svs_patches_dir, mask_patches_dir)
            slide.close()

            processed_svs_files += 1
            total_patches += len(patches)

            print(f"Processed SVS file {processed_svs_files}/{total_svs_files}: {base_name}. Extracted {len(patches)} patches.")

    return processed_svs_files, total_patches
