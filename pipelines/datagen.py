"""
Data generator
Takes images and labels
"""
import os
import torch
import rasterio
import numpy as np
from glob import glob
import kornia.augmentation as K
from rasterio.features import rasterize
from kornia.geometry import vflip, hflip
from kornia.geometry.transform import Rotate


class Datagen:
    def __init__(self, data_dir,
                 set_name='train',
                 rescale_value=2000,
                 patch_size=256,
                 stretch_setting=2,
                 random_crop=False):

        self.data_dir = data_dir
        self.set_name = set_name
        self.rescale_value = rescale_value
        self.random_crop = random_crop
        self.image_paths = sorted(glob(os.path.join(self.data_dir, self.set_name, "images/*.tif")))
        self.label_paths = sorted(glob(os.path.join(self.data_dir, self.set_name, "labels/*.tif")))
        if self.set_name == 'both':
            self.image_paths = sorted(glob(os.path.join(self.data_dir, 'train', "images/*.tif")))
            self.label_paths = sorted(glob(os.path.join(self.data_dir, 'train', "labels/*.tif")))
            self.image_paths += sorted(glob(os.path.join(self.data_dir, 'val', "images/*.tif")))
            self.label_paths += sorted(glob(os.path.join(self.data_dir, 'val', "labels/*.tif")))
        self.patch_size = patch_size
        self.stretch_setting = stretch_setting

        # augmentation chances
        self.hflip_chance = 0.5
        self.vflip_chance = 0.5
        self.rotate_chance = 0.5
        self.colorjitter_chance = 0.25
        self.brightness_chance = 0.25
        self.gaussblurr_chance = 0.10
        self.gaussnoise_chance = 0.10
        self.sharpness_chance = 0.10

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with rasterio.open(self.image_paths[index]) as src:
            x = src.read(1).astype(np.float32) / self.rescale_value
            x = np.expand_dims(x, 0)

        with rasterio.open(self.label_paths[index]) as src:
            y = src.read()
            inner = np.where(y==1, 1, 0)
            outer = np.where(y==2, 1, 0)
            y = np.concatenate((inner, outer), axis=0)

        image = torch.from_numpy(x).float()
        label = torch.from_numpy(y).long()
        filename = os.path.basename(self.image_paths[index])

        # random crop of image and annotations
        if self.random_crop and self.patch_size != 512:
            cx = np.random.randint(0, image.shape[1] - self.patch_size)
            cy = np.random.randint(0, image.shape[2] - self.patch_size)
            image = image[:, cy:cy + self.patch_size, cx:cx + self.patch_size]
            label = label[:, cy:cy + self.patch_size, cx:cx + self.patch_size]

            # apply image stretch
            if self.stretch_setting == 1:
                image = image * np.random.uniform(0.8, 1.25)
            elif self.stretch_setting == 2:
                image = image * np.random.uniform(0.5, 2.0)

        return image, label, filename

