import os
import csv
import torch
import rasterio
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from itertools import product
from rasterio.windows import Window
from tools import network, network_elu, network_elu_bn


# function that take dictionary of models and gives the final mask
@torch.no_grad()
def ensemble_union_predict(models,
                           input_patch,
                           ensemble_mode="majority"):
    """
    Predict from multiple models
    """

    individual_masks = []

    for model in models.values() if isinstance(models, dict) else models:
        logits = model(input_patch)

        # Handle (B, H, W) → (B, 1, H, W)
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        probs = torch.sigmoid(logits)

        # binary_mask = (probs > 0.5).to(torch.uint8)
        individual_masks.append(probs)

    stacked = torch.stack(individual_masks, dim=0)
    if ensemble_mode == "majority":
        # MAJORITY
        mask = torch.mean(stacked, dim=0)
        mask = (mask > 0.5).to(torch.uint8)
        # mask = (mask * 100).to(torch.uint8)
    elif ensemble_mode == "intersection":
        # INTERSECTION
        mask = stacked > 0.5
        mask = torch.all(mask, dim=0).to(torch.uint8)
    elif ensemble_mode == "atleast1":
        # ATLEAST ONE
        mask = stacked > 0.5
        mask = (torch.sum(mask, dim=0) >= 1).to(torch.uint8)
    elif ensemble_mode == "atleast2":
        # ATLEAST TWO
        mask = stacked > 0.5
        mask = (torch.sum(mask, dim=0) >= 2).to(torch.uint8)
    elif ensemble_mode == "atleast3":
        # ATLEAST THREEE
        mask = stacked > 0.5
        mask = (torch.sum(mask, dim=0) >= 3).to(torch.uint8)
    elif ensemble_mode == "atleast4":
        # ATLEAST FOUR
        mask = stacked > 0.5
        mask = (torch.sum(mask, dim=0) >= 4).to(torch.uint8)
    else:
        raise NotImplementedError(f"ensemble_mode={ensemble_mode} not implemented")

    return mask


def get_patch_offsets(width, height, stride):
    """
    Get a list of patch offsets based on image size, patch size and stride.
    """
    # Create iterator of all patch offsets, as tuples (x_off, y_off)
    patch_offsets = list(product(range(0, width, stride), range(0, height, stride)))
    return patch_offsets


# function that takes the image and give the prediction
def apply_model_on_geotiff(
        geotiff_path,
        checkpoint_path,
        output_path,
        device='cpu',
        rescale_value = 2000,
        patch_size=None,
        border=47,
        ensemble_mode='majority',
        model_name='unet_elu'):

    models = {}
    for i, path in enumerate(checkpoint_path):
        # Setup model
        if model_name == 'unet_elu':
            model = network_elu.TinyUNet()
        elif model_name == 'unet_elu_bn':
            model = network_elu_bn.TinyUNet()
        elif model_name == 'unet_relu_bn':
            model = network.TinyUNet()
        else:
            raise Exception("Unknown model name")
        # model = network.TinyUNet()
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()
        models[f"model_{i + 1}"] = model

    if patch_size is not None:
        img = rasterio.open(geotiff_path, tiled=True, blockxsize=256, blockysize=256)
        profile = img.profile.copy()
        h, w = img.height, img.width
        stride = patch_size - 2 * border
        offsets = get_patch_offsets(w, h, stride)

        # offset stats for identifying border patch
        max_col = max(c for c, r in offsets)
        max_row = max(r for c, r in offsets)

        big_window = Window(0, 0, w, h)
        output_mask = np.zeros((h, w), dtype=np.uint8)

        # loop through patch size and predict
        for i, (col_off, row_off) in enumerate(
                tqdm(offsets, desc="Predicting patches")):
            # prepare the patch
            patch_window = Window(col_off=col_off,
                                  row_off=row_off,
                                  width=patch_size,
                                  height=patch_size).intersection(big_window)
            patch_img = img.read([1], window=patch_window).astype(np.float32) / rescale_value
            current_h, current_w = patch_img.shape[1:]
            padded_patch = np.zeros((1, patch_size, patch_size), dtype=np.float32)
            padded_patch[:, :current_h, :current_w] = patch_img

            patch = torch.from_numpy(padded_patch)
            patch = patch.unsqueeze(0).to(device)

            # use reflection padding if the patch is at border
            if row_off in [0, max_row] or col_off in [0, max_col]:
                patch = nn.ReflectionPad2d(47)(patch)
            # predict using retrained models
            if len(checkpoint_path) > 1:
                pred = ensemble_union_predict(models, patch, ensemble_mode=ensemble_mode)
            else:
                model = models['model_1']
                logits = model(patch)
                probs = torch.sigmoid(logits) * 100
                pred = probs.to(torch.uint8)

            # replace in the output mask
            pred = pred.squeeze().cpu().numpy()
            valid_h = current_h - (2 * border)
            valid_w = current_w - (2 * border)
            start_row = row_off + border
            start_col = col_off + border

            # deal with border patches
            if row_off in [0, max_row] or col_off in [0, max_col]:
                pred = pred[1:-1, 1:-1]
                p_row_start, p_row_end, p_col_start, p_col_end = 47, -47, 47, -47
                if col_off == 0:
                    valid_w += border
                    start_col -= border
                    p_col_start = 0
                if col_off == max_col:
                    valid_w += border
                    p_col_end = -1
                if row_off == 0:
                    valid_h += border
                    start_row -= border
                    p_row_start = 0
                if row_off == max_row:
                    valid_h += border
                    p_row_end = -1
                pred = pred[p_row_start:p_row_end, p_col_start:p_col_end]

            end_row = start_row + valid_h
            end_col = start_col + valid_w

            output_mask[
                start_row:end_row,
                start_col:end_col
            ] = pred[:valid_h, :valid_w]
        img.close()
    else:

        with rasterio.open(geotiff_path) as src:
            patch_img = src.read([1]).astype(np.float32) / rescale_value
            profile = src.profile.copy()
            # output_mask = np.zeros((src.height, src.width), dtype=np.uint8)
        patch = torch.from_numpy(patch_img)
        patch = nn.ReflectionPad2d(47)(patch)
        patch = patch.unsqueeze(0).to(device)
        if len(checkpoint_path) > 1:
            pred = ensemble_union_predict(models, patch, ensemble_mode=ensemble_mode)
        else:
            raise NotImplementedError
        pred = pred.squeeze().cpu().numpy()
        # output_mask[47:-47, 47:-47] = pred
        output_mask = pred[1:-1, 1:-1]

    profile.update(count=1,
                   dtype=rasterio.uint8,
                   compress='deflate',
                   tiled=True,
                   BIGTIFF='YES',
                   nodata=0
                   )


    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output_mask, 1)


class ArcticPredict:
    def __init__(self, **kwargs):
        # self.images = sorted(glob(os.path.join(kwargs['image_dir'], '*.tif')))
        self.images = kwargs['images']
        self.pretrained_model = kwargs['pretrained_model']
        self.ensemble = kwargs['ensemble']
        self.out_dir = kwargs['out_dir']
        self.patch_size = kwargs['patch_size']
        self.ensemble_mode = kwargs['ensemble_mode']
        self.model_name = kwargs['model_name']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.csv_path = kwargs['csv_path']

    def run(self):
        for image_path in self.images:
            print(f"Processing {image_path}")
            output_path = os.path.join(self.out_dir, os.path.basename(image_path))
            try:
                apply_model_on_geotiff(geotiff_path=image_path,
                                       checkpoint_path=self.pretrained_model,
                                       output_path=output_path,
                                       device=self.device,
                                       patch_size=self.patch_size,
                                       ensemble_mode=self.ensemble_mode,
                                       model_name=self.model_name)
                status = 'success'
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
                status = 'fail'

            print(f'Finished processing and saved to {output_path}')

            # Append result to CSV after each image
            file_exists = os.path.isfile(self.csv_path)
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['image_path', 'status'])
                if not file_exists:
                    writer.writeheader()
                writer.writerow({'image_path': image_path, 'status': status})


