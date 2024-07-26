import os
import json
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape
from openslide import OpenSlide
from affine import Affine

def process_files(svs_folder, geojson_folder, output_folder):
    masks_info = []  # To store information about processed masks
    for svs_file in os.listdir(svs_folder):
        if svs_file.endswith('.svs'):
            svs_path = os.path.join(svs_folder, svs_file)
            geojson_path = os.path.join(geojson_folder, os.path.splitext(svs_file)[0] + '.geojson')

            wsi = OpenSlide(svs_path)

            with open(geojson_path, "r") as f:
                geojson_data = json.load(f)

            if not isinstance(geojson_data, list):
                geojson_data = [geojson_data]

            output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(geojson_path))[0] + '_mask.tif')

            image_width, image_height = wsi.level_dimensions[0]
            transform = Affine(wsi.level_downsamples[0], 0, 0, 0, wsi.level_downsamples[0], 0)

            def convert_coordinates(coordinates):
                return [(int((y - wsi.properties['hamamatsu.origin_y']) / wsi.level_downsamples[0]), int((x - wsi.properties['hamamatsu.origin_x']) / wsi.level_downsamples[0])) for x, y in coordinates]

            mask = rasterize(
                [shape(feature["geometry"]) for feature in geojson_data],
                out_shape=(image_height, image_width),
                transform=transform,
                all_touched=True,
                fill=0,
                default_value=255,
                dtype=np.uint8
            )

            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                width=image_width,
                height=image_height,
                count=1,
                dtype=rasterio.uint8,
                transform=transform,
            ) as dst:
                dst.write(mask, 1)

            masks_info.append({'name': os.path.splitext(os.path.basename(geojson_path))[0], 'mask': mask})
            wsi.close()


# Confirm that all svs files have a corresponding binary mask
svs_files = [f for f in os.listdir(svs_folder) if f.endswith('.svs')]
num_svs_files = len(svs_files)

mask_files = [f for f in os.listdir(output_folder) if f.endswith('_mask.tif')]
num_mask_files = len(mask_files)

processed_svs_files = [f for f in os.listdir(destination_folder) if f.endswith('.svs')]
processed_svs_basenames = {os.path.splitext(f)[0] for f in processed_svs_files if f.endswith('.svs')}
mask_basenames = {os.path.splitext(f)[0].replace('_mask', '') for f in mask_files}

print(f'Number of SVS files: {num_svs_files}')
print(f'Number of binary masks: {num_mask_files}')

processed_svs_without_masks = processed_svs_basenames - mask_basenames
masks_without_svs = mask_basenames - processed_svs_basenames

print("SVS files without corresponding masks:")
for svs in processed_svs_without_masks:
    print(svs)

print("\nMasks without corresponding SVS files:")
for mask in masks_without_svs:
    print(mask)
