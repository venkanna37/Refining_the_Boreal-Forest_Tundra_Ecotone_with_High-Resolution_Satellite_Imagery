⚠️ **This code is under preparation. It will be ready soon!**

# Refining the Boreal-Forest Tundra Ecotone with High-Resolution Satellite Imagery
This repository contains the training U-Net and other essential codes for mapping trees in Tundra region from high-resolution panchromatic images.

[![DOI](https://zenodo.org/badge/21889707.svg)](https://zenodo.org/badge/latestdoi/21889707)

The code is structured as separate python file for each task  (e.g., `train.py` for training).
Each python file contains a considerable part of the pipeline and they are supported with other python files available in the `tools` directory.

The outline of this repository is given as follows:
* [**Installation**](#installation)
* [**Data**](#data)
* [**Training**](#training)
* [**Evaluation**](#evaluation)
* [**Predict tiles**](#predict-tiles)
* [**Postprocessing**](#postprocessing)


## Installation
Installing all the packages listed in `requirements.txt` allows you to run the code.
The commands below create Conda environment and install all the packages.
You can also install packages in other python environments instead of creating conda environment.

```bash
conda create -n "arctic" python=3.11.0
pip install -r requirements.txt
```

---

## Data
Custom datasets must follow the folder structure shown below.
Both images and labels should prepare and save in `.tif` format in respective folder.
Example data of building segmentation (not trees) are provided in the `datasets/sample_data` folder.
The training pipeline saves checkpoints in `runs` folder as shown below.
Custom directories `datasets` and `runs` folders can be specified  while running training and evaluation scripts.

```
datasets                  # All datasets or dataset folder
└── fold1                 # Complete dataset
    ├── train             # Training set
    │   ├── images        # Input images in .tif format
    │   └── labels        # Input labels in .tif format
    ├── val               # Validation set
    │   └── ...           # Same structure as train
    └── test              # Test set
        └── ...           # Same structure as train
└── sample_dataset        # sample dataset for testing the code
        └── ...           # Same structure as fold1
└── tiles_folder          # Folder with all bigger tiles

runs                      # All checkpoints or trained models
└── unet_fold1            # checkpoint with --keyword unet_fold1
    ├── best_f1score.pt   # Model weights with highest f1-score on val set
    └── tnet_precision.pt # Model weights with highest precision on val set
```

---

## Training
To train the model on `sample_data`, run:

```bash
python train.py --keyword unet_fold1 --data_dir ./datasets/sample_data
```

Change `--data_dir` to custom data directory to train on it. Also change `--keyword` to save checkpoints to new folder.

For all available options like `--data_dir` and `--keyword`, run:

```bash
python train.py --help
```

---

## Evaluation
To evaluate the model on `sample_data` (on `test` set), run:

```bash
python eval.py --keyword unet_fold1 --data_dir ./datasets/sample_data --set_name test
```

Change `--keyword`, `--data_dir` and `--set_name` accordingly and run `python pred.py --help` for all options.


## Predict tiles
To predict all tiles in the folder `tiles_folder`, run:

```bash
python pred.py --keyword test_run --data_dir ./datasets/tiles_folder
```

Change `--keyword` and `--data_dir` accordingly and run `python pred.py --help` for all options.

## Postprocessing
Coming soon

## A note on the data source
This code is customized for the data that was available to us.
It relies on satellite images with single channel and assumes that the one channel is stored in independent file
and consequently read independently.
In case your data is available in a single file (maybe with multiple channels), then use `--num_channels` options.