## CycleGAN-Driven Data Augmentation for Improved Neural Network Disease Detection 


Please find and download the BRACS Dataset from here: https://www.bracs.icar.cnr.it/ 

PS: This dataset is only for practice as we have trained our model using a private institutional dataset from Nagasaki University Hospital. However, please feel free to use the BRACS dataset to validate our method as it is publicly available :)

After downloading, please open the SVS files and their corresponding .qpdata annotations using the QuPath software, which you can install from here: https://qupath.github.io/

Then, under the File tab, choose the Export objects as GeoJSON feature and select the Export as FeatureCollection option to create a .geojson file that you can use to create binary masks. Training our CycleGAN requires both an RGB image and a grayscale binary mask.

Also, since each SVS image is too large to process at once, we have extracted patches from each image and its corresponding binary mask (1024*1024) for training. 

This repository contains scripts to help you:

1. Generate binary masks from SVS images guided by coordinates in the .geojson file

2. Extract 1024*1024 .png patches from SVS images + TIFF binary masks (class: with lesions) guided by coordinates in the .geojson file

3. Extract 1024*1024 .png patches from SVS images (class: without lesions) with randomness conditionally applied to avoid extracting too many patches

4. Train a CycleGAN model to synthesize pathology onto healthy images with binary masks as conditional input

5. Evaluate the CycleGAN model using IoU and SSIM metrics 

6. Train a CycleGAN model to synthesize pathology onto healthy images without any conditional input

7. Add synthetic images (created from a CycleGAN trained with binary masks) to the original training dataset for a classification task to evaluate whether fake images improve a neural network model's generalization abilities 

8. Add synthetic images (created from a CycleGAN trained without binary masks) to the original training dataset for a classification task to evaluate whether fake images improve a neural network model's generalization abilities 

9. Train 4 independent sets of models and measure the sensitivity of models trained with real data only, synthetic data only, real + synthetic data for fake images created from a CycleGAN trained with binary masks, and real + synthetic data for fake images created from a CycleGAN trained without binary masks

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
@inproceedings{sice23,
  author = {{Annette Waithira Irungu} and {Kotaro Sonoda} and {Makoto Kawamoto} and {Kris Lami} and {Junya Fukuoka} and {Senya Kiyasu}},
  title = {Enhancing mycobacterial disease detection with generative image translation},
  booktitle = {SPIE/COS Photonics Asia},
  year = {2024},
  month = {oct},
  address = {Nantong, Jiangsu, China}
}
```
