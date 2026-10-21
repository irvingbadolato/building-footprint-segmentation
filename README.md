

# Building Footprint Segmentation with RefiNet

This repository extends [Daniel](https://github.com/Daniel-ime/building-footprint-segmentation-resnet50)’s proposal for segmenting building footprints from aerial imagery using the RefineNet model. A new set of experiments is proposed, focusing on the same study areas of interest to the Pereira Passos Municipal Institute of Urban Planning (IPP). The cropping methodology has been revised, and the dataset for the municipality of Rio de Janeiro has undergone curation.

## Overview

This project uses advanced segmentation techniques to identify and extract building footprints from overhead imagery, which is useful for urban planning, mapping applications, and geospatial analysis.

## Requirements

-   Python 3.12+
-   CUDA-compatible GPU (NVIDIA) or ROCm-compatible GPU (AMD)
-   Git

### Key Dependencies

-   **PyTorch** (ROCm or CUDA version depending on your GPU)
-   **building-footprint-segmentation** 0.2.4
-   **NumPy** 1.26.4 (specific version required)
-   **albumentations** 2.0.8
-   **opencv-python-headless** 4.11.0.86
-   **torchvision**, **torchaudio**
-   **scikit-learn**, **scipy**
-   **matplotlib**, **Pillow**
-   **Jupyter** (for notebooks)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/irvingbadolato/building-footprint-segmentation
```

### 2. Create Conda Environment

```bash
conda create -n your_env_for_seg python=3.12
conda activate your_env_for_seg

```

### 3. Install PyTorch

Refer to the [get started](https://pytorch.org/get-started/locally/) section on the official PyTorch website and select a command to install the stable version via pip, based on your hardware configuration and operating system. Keep in mind that this project is developed in Python.

### 4. Install Other Required Packages

**Critical:** NumPy must be version 1.26.4:

```bash
pip install numpy==1.26.4 albumentations opencv-python-headless scikit-learn scipy matplotlib Pillow jupyter jupyterlab ipywidgets tqdm pyyaml
```


### 5. Install Building Footprint Segmentation
Implementation was done in a way that it is necessary to have the main folder inside your python environment. You can do it by pip install on the right folder as there is a setup.py file. 

```bash
pip install building-footprint-segmentation
```

However if it doesn't work simply copy paste it to the environment folder.

## Project Structure

To reproduce the experiments in this project, the folder structure of the dataset is highly relevant; here is an example of the expected structure for your training and evaluation data:
```
.
├── data/rio/building_patches/                # Dataset directory, this is the expected
   ├──train/								  # format, other formats wont work.
      ├──images
      ├──labels       
	├──val/
      ├──images
      ├──labels
	├──test/
      ├──images
      ├──labels
```

This project uses GDAL; therefore, image formats suitable for spatial data are supported. Only the first three bands of each image will be used. This approach was adopted to allow for the use of a ResNet encoder pre-trained on ImageNet data.
All models in this set of experiments perform binary segmentation. Therefore, data normalization assumes that pixels with non-zero values ​​in any of the label channels are to be considered as the object. In other words, the background must be labeled as zero.

## Usage

All the experiments in the examples use an environment variable to configure the local project path. Before using the provided scripts, it is recommended to update this variable in the `environment.py` file. All YAML configuration files also assume that the base paths for the dataset and for log storage have been updated.

### Dataset preprocessing

In the "dataset crop" notebook, the preprocessing operations performed on the Rio de Janeiro dataset were detailed. The orthomosaics in the Rio de Janeiro dataset are quite large (approximately 15k rows by 21k columns). The approach taken involves, in short, dividing these images into smaller, non-overlapping blocks, which are then partitioned into training, validation, and test sets. These blocks are further subdivided into patches, with their size determined by the network's input dimensions (384x384 pixels); in this step, the cropping process allows for overlap. Patches containing only a small number of pixels representing buildings are discarded. The assignment of each block to a specific set (training, validation, or test) is based on the number of patches that remain after the discarding process. By separating the data at the block level rather than the patch level, the spatial adjacency between the datasets is minimized, which facilitates a more robust evaluation of the models' generalizability.

### Run experiments

All other notebooks in the examples directory assume that a dataset - following the proposed structure - has been preprocessed, and that the configuration files and the environment variable have been set up beforehand. A script for coordinating the execution of the proposed experiments can be found in `run_experiments.py` file. Running it without modifications results in 3 rounds of training and evaluation for 8 proposed experimental configurations.
