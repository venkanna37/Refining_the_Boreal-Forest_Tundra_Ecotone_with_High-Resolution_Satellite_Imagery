# Refining the Boreal-Forest Tundra Ecotone with High-Resolution Satellite Imagery
This repository contains the neural network model (UNet) and other essential codes for segmenting trees in Tundra region from high-resolution panchromatic images.

The code is structured as separate python file for each task  (e.g., `train.py` for training).
Each python file contains a considerable part of the pipeline and they are supported with other python files available in the tools directory.

[![DOI](https://zenodo.org/badge/21889707.svg)](https://zenodo.org/badge/latestdoi/21889707)

* [**Installation**](#installation)
* [**Data**](#data)
* [**Training**](#training)
* [**Evaluation**](#evaluation)
* [**Predict tiles**](#predict-tiles)
* [**Postprocessing**](#postprocessing)


## Installation
Installing the packages in `requirements.txt` allows using all provided code.

```bash
conda create -n "arctic" python=3.11.0
pip install -r requirements.txt
```

---

## Data
Custom datasets should follow the datasets folder structure below.
Both images and labels should be prepared before and save in `.tif` format.
Some example dataset from similar task added in `datasets/sample_data` folder.
The training pipeline save checkpoints in runs folder.
Both custom `datasets` and `runs` folders can be specified during training and evaluation.

```
datasets                  # All datasets
└── fold1                 # Complete dataset
    ├── train             # Training set
    │   ├── images        # Input images in tiff format
    │   └── labels        # Input labels in tiff format
    ├── val               # Validation set
    │   └── ...           # Same structure as train
    └── test              # Test set
        └── ...           # Same structure as train
└── sample_dataset        # Complete dataset
        └── ...           # Same structure as fold1
└── tiles_folder          # Folder with all tiles

runs                      # All checkpoints or trained models
└── unet_fold1            # checkpoint with --keyword unet_fold1
    ├── best_f1score.pt   # Model weights with highest f1-score on val set
    └── tnet_precision.pt # Model weights with highest f1-score on val set
```

---

## Training
To train the model on `sample_data`, run:

```bash
python train.py --keyword test_run --data_dir ./datasets/sample_data
```

Change `--data_dir` to custom data directory to train on it.

For all available options like `--data_dir`, run:

```bash
python train.py --help
```

---

## Evaluation
To evaluate the model on `sample_data` (on `test` set), run:

```bash
python eval.py --keyword test_run --data_dir ./datasets/sample_data --set_name test
```

Change `--keyword`, `--data_dir` and `--set_name` accordingly and try `python pred.py --help` for the options.


## Predict tiles
To predict all tiles in folder `tiles_folder`, run:

```bash
python pred.py --keyword test_run --data_dir ./datasets/tiles_folder
```

Change `--keyword` and `--data_dir` accordingly and try `python pred.py --help` for the options.

## Postprocessing
Coming soon

## A note on the data source
This code is customized for the data that was available to us.
It relies on satellite images with single channel and assumes that the one channel is stored in independent file
and consequently read independently.
In case your data is available in a single file (maybe with multiple channels), then use `--num_channels` options.