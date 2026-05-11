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
    def __init__(self, data_dir, set_name='train', rescale_value=2000,
                 patch_size=256, geom_aug=False):
        self.data_dir = data_dir
        self.set_name = set_name
        self.rescale_value = rescale_value
        self.image_paths = sorted(glob(os.path.join(self.data_dir, self.set_name, "images/*.tif")))
        self.label_paths = sorted(glob(os.path.join(self.data_dir, self.set_name, "labels/*.tif")))
        self.patch_size = patch_size
        self.geom_aug = geom_aug

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

        # random crop of image and annotations
        if self.set_name == 'train' and self.patch_size != 512:
            cx = np.random.randint(0, image.shape[1] - self.patch_size)
            cy = np.random.randint(0, image.shape[2] - self.patch_size)
            image = image[:, cy:cy + self.patch_size, cx:cx + self.patch_size]
            label = label[:, cy:cy + self.patch_size, cx:cx + self.patch_size]

        # apply image stretch

        return image, label

    def augment(self, X, y, device):
        B, C, H, W = X.shape
        y = y.float()

        if self.geom_aug:
            # flipping horizontal
            hflip_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.hflip_chance)
            X = X * (1 - hflip_coin) + hflip(X) * hflip_coin
            y = y * (1 - hflip_coin) + hflip(y) * hflip_coin

            # flipping vertical
            vflip_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.vflip_chance)
            X = X * (1 - vflip_coin) + vflip(X) * vflip_coin
            y = y * (1 - vflip_coin) + vflip(y) * vflip_coin

            # Rotation 90
            rot90_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.rotate_chance)
            augX = K.RandomRotation90(times=(1, 1), p=1, resample='bilinear', keepdim=True)
            augy = K.RandomRotation90(times=(1, 1), p=1, resample='nearest', keepdim=True)
            X = X * (1 - rot90_coin) + augX(X) * rot90_coin
            y = y * (1 - rot90_coin) + augy(y) * rot90_coin

            # Rotation 180
            rot180_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.rotate_chance)
            augX = K.RandomRotation90(times=(2, 2), p=1, resample='bilinear', keepdim=True)
            augy = K.RandomRotation90(times=(2, 2), p=1, resample='nearest', keepdim=True)
            X = X * (1 - rot180_coin) + augX(X) * rot180_coin
            y = y * (1 - rot180_coin) + augy(y) * rot180_coin

            # Rotation 270
            rot270_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.rotate_chance)
            augX = K.RandomRotation90(times=(3, 3), p=1, resample='bilinear', keepdim=True)
            augy = K.RandomRotation90(times=(3, 3), p=1, resample='nearest', keepdim=True)
            X = X * (1 - rot270_coin) + augX(X) * rot270_coin
            y = y * (1 - rot270_coin) + augy(y) * rot270_coin

        # random erasing
        # erasing_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.erasing_chance)
        # EraseT = K.RandomErasing(scale=(0.005, 0.05), ratio=(0.3, 3.3), value=0.0, p=1, keepdim=True)
        # X = X * (1 - erasing_coin) + EraseT(X) * erasing_coin
        # y = y * (1 - erasing_coin) + EraseT(y) * erasing_coin

        # Brightness -> per images: Changes brightness between 0.8 and 1.2
        # colorjitter_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.colorjitter_chance)
        # aug = K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0, p=1)
        # X = X * (1 - colorjitter_coin) + aug(X) * colorjitter_coin

        brightness_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.brightness_chance)
        X = X * (1 - brightness_coin) + (X + (torch.rand(size=(B, 1, 1, 1), device=device) * 0.3 - 0.15)) * brightness_coin

        gaussblurr_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.gaussblurr_chance)
        aug = K.RandomGaussianBlur(kernel_size=(3, 7), sigma=(0, 0.3), p=1)
        X = X * (1 - gaussblurr_coin) + aug(X) * gaussblurr_coin

        gaussnoise_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.gaussnoise_chance)
        aug = K.RandomGaussianNoise(mean=0., std=0.05, p=1)
        X = X * (1 - gaussnoise_coin) + aug(X) * gaussnoise_coin

        sharpness_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.sharpness_chance)
        aug = K.RandomSharpness(sharpness=0.5, p=1)
        X = X * (1 - sharpness_coin) + aug(X) * sharpness_coin

        return X, y

    def augment_modified(self, X, y, device):
        B, C, H, W = X.shape
        y = y.float()

        # flipping horizontal
        # hflip_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.hflip_chance)
        # X = X * (1 - hflip_coin) + hflip(X) * hflip_coin
        # y = y * (1 - hflip_coin) + hflip(y) * hflip_coin
        #
        # # flipping vertical
        # vflip_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.vflip_chance)
        # X = X * (1 - vflip_coin) + vflip(X) * vflip_coin
        # y = y * (1 - vflip_coin) + vflip(y) * vflip_coin

        # Brightness -> per images: Changes brightness between 0.8 and 1.2
        colorjitter_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.colorjitter_chance)
        aug = K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0, p=1)
        X = X * (1 - colorjitter_coin) + aug(X) * colorjitter_coin

        gaussblurr_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.gaussblurr_chance)
        aug = K.RandomGaussianBlur(kernel_size=(3, 7), sigma=(0, 0.3), p=1)
        X = X * (1 - gaussblurr_coin) + aug(X) * gaussblurr_coin

        # gaussnoise_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.gaussnoise_chance)
        # aug = K.RandomGaussianNoise(mean=0., std=0.05, p=1)
        # X = X * (1 - gaussnoise_coin) + aug(X) * gaussnoise_coin

        sharpness_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.sharpness_chance)
        aug = K.RandomSharpness(sharpness=0.5, p=1)
        X = X * (1 - sharpness_coin) + aug(X) * sharpness_coin

        # # affine transformation
        # rot_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.rotate_chance)
        # angles = (torch.rand(B, device=device) * 2 - 1) * 180
        # augX = Rotate(angle=angles, mode='bilinear')
        # augy = Rotate(angle=angles, mode='nearest')
        # X = X * (1 - rot_coin) + augX(X) * rot_coin
        # y = y * (1 - rot_coin) + augy(y) * rot_coin

        return X, y

