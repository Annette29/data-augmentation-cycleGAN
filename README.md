## CycleGAN-Driven Data Augmentation for Improved Neural Network Disease Detection 


Please find and download the BRACS Dataset from here: https://www.bracs.icar.cnr.it/ 

After downloading, please open the SVS files and their corresponding .qpdata annotations using the QuPath software, which you can install from here: https://qupath.github.io/

Then, under the File tab, choose the Export objects as GeoJSON feature and select the Export as FeatureCollection option to create a .geojson file that you can use to create binary masks. Training our CycleGAN requires both an RGB image and a grayscale binary mask.

Also, since each SVS image is too large to process at once, we have extracted patches from each image and its corresponding binary mask (1024*1024) for training. 

This repository contains scripts to help you:

1. Generate binary masks from SVS images guided by coordinates in the .geojson file

2. Extract 1024*1024 .png patches from SVS images + TIFF binary masks (class: with lesions) guided by coordinates in the .geojson file

3. Extract 1024*1024 .png patches from SVS images (class: without lesions) with randomness conditionally applied to avoid extracting too many patches

4. Train a CycleGAN model to synthesize pathology onto healthy images with binary masks as conditional input

5. Add synthetic images to the original training dataset for a classification task to evaluate whether fake images improve a neural network model's generalization abilities 

8. Train 6 model instances and measure the sensitivity of models trained with real data only, synthetic data only, real + synthetic data with and without classical data augmentation methods

---

To use our project, please follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Annette29/data-augmentation-cycleGAN.git
   ```

2. **Navigate to the project directory:**
   ```bash
   cd data_augmentation_using_cycleGAN
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

Ensure you have `git` and `pip` installed on your system before running these commands. If you encounter any issues with permissions, consider using a virtual environment or adding `--user` to the `pip install` command.

---
Citation
If you use this code or find our work helpful, please consider citing this paper:

```bibtex
@inproceedings{spie24,
  author = {{Annette Waithira Irungu} and {Kotaro Sonoda} and {Makoto Kawamoto} and {Kris Lami} and {Junya Fukuoka} and {Senya Kiyasu}},
  title = {CycleGAN-driven data augmentation for improved neural network disease detection},
  booktitle = {SPIE/COS Photonics Asia},
  year = {2024},
  month = {oct},
  address = {Nantong, Jiangsu, China}
}
```
