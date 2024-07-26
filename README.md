## CycleGAN-Driven Data Augmentation for Improved Neural Network Disease Detection 


Please find and download the BRACS Dataset from here: https://www.bracs.icar.cnr.it/ 

After downloading, please open the SVS files and their corresponding .qpdata annotations using the QuPath software (https://qupath.github.io/)

Then, under the File tab, choose the Export objects as GeoJSON feature, and select the Export as FeatureCollection option to create a .geojson file that you can then use to create binary masks, as training our CycleGAN requires both an RGB image and a grayscale binary mask.

Also, since each SVS image is too large to process at once, we have extracted patches from each image and its corresponding binary mask (1024*1024) for training. 

This repository contains scripts for:

1. Generating binary masks from SVS images guided by coordinates in the .geojson file

2. Extracting 1024*1024 .png patches from SVS images + TIFF binary masks (class: with lesions) guided by coordinates in the .geojson file

3. Extracting 1024*1024 .png patches from SVS images (class: without lesions) with randomness conditionally applied to avoid extracting too many patches


