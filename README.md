# Refining the Boreal-Forest Tundra Ecotone with High-Resolution Satellite Imagery
This repository contains the training code for a U-Net model, along with other essential code,
for mapping trees in tundra regions from high-resolution panchromatic satellite imagery.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21892162.svg)](https://doi.org/10.5281/zenodo.21892162)

The code is organized as a separate Python file for each task (e.g., `train.py` for training).
Each file implements a substantial part of complete pipeline and is supported by helper modules in the `tools` directory.

The outline of this repository is given as follows:
* [**Installation**](#installation)
* [**Data**](#data)
* [**Training**](#training)
* [**Evaluation**](#evaluation)
* [**Predict tiles**](#predict-tiles)
* [**Postprocessing**](#postprocessing)


## Installation
Installing the packages listed in `requirements.txt` is all that's required to run the code.
The commands below create a Conda environment and install the dependencies.
The packages can also be installed into any other Python environment if you prefer not to use Conda.

```bash
conda create -n "arctic" python=3.11.0
pip install -r requirements.txt
```

---

## Data
Custom datasets must follow the folder structure shown below.
Both images and labels should be prepared and save in `.tif` format in respective folder.
The training pipeline saves checkpoints in `runs` folder as shown below.
`tiles_folder` is the directory with larger tiles.
Custom directories of `datasets`, `tiles_folder` and `runs` folders can be specified  while running training and evaluation scripts.

```
datasets                  # All datasets or dataset folder
└── arctic_data           # Complete dataset
    ├── train             # Training set
    │   ├── images        # Input images in .tif format
    │   └── labels        # Input labels in .tif format
    ├── val               # Validation set
    │   └── ...           # Same structure as train
    └── test              # Test set
        └── ...           # Same structure as train
└── tiles_folder          # Folder with all bigger tiles

runs                      # All checkpoints or trained models
└── unet_fold1            # checkpoint with --keyword unet_fold1
    ├── best_f1score.pt   # Model weights with highest f1-score on val set
    └── tnet_precision.pt # Model weights with highest precision on val set
```

---

## Training
To train the model on `arctic_data`, run:

```bash
python train.py --keyword arctic_data_run --data_dir ./datasets/arctic_data/
```

Change `--data_dir` to point to a custom dataset, and `--keyword` to save checkpoints under a new name.

For the full list of options (e.g., `--data_dir`, `--keyword`), run:

```bash
python train.py --help
```

---

## Evaluation
To evaluate the model on `arctic_data` (on `test` set), run:

```bash
python eval.py --keyword arctic_data_run --data_dir ./datasets/arctic_data/ --set_name test
```

Adjust `--keyword`, `--data_dir`, and `--set_name` as needed. For the full list of options, run: `python pred.py --help`


## Predict tiles
To run inference on all tiles in `tiles_folder`, run:

```bash
python pred.py --keyword arctic_data_run --in_dir ./datasets/tiles_folder --out_dir ./runs/predictions/
```

Adjust `--keyword`, `--in_dir`, and `--out_dir` as needed. For the full list of options, run: `python pred.py --help`

## Postprocessing
Use `computeTreeCanopyCover.py` and `identifyTreeline.py` to estimate tree canopy cover and the treeline, respectively.
Running these scripts requires `gdal` to be installed that is not added in `requirements.txt` file.

## A note on the data source
All default hyperparameters in train.py (e.g., learning rate, patch size) were selected for our specific dataset.
The evaluation and prediction code is also written for a fixed patch/image size,
so parts of the code may need to be adjusted if the patch size changes.