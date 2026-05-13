import os
import torch
import rasterio
import numpy as np
from glob import glob
from tqdm import tqdm
from pipelines import network
from itertools import product
from rasterio.windows import Window


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
    # MAJORITY
    if ensemble_mode == "majority":
        mask = torch.mean(stacked, dim=0)
        mask = (mask > 0.5).to(torch.uint8)
    elif ensemble_mode == "intersection":
        # INTERSECTION
        mask = stacked > 0.5
        mask = torch.all(mask, dim=0).to(torch.uint8)

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
        patch_size=1024,
        border=47):

    models = {}
    for i, path in enumerate(checkpoint_path):
        model = network.TinyUNet()
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()
        models[f"model_{i + 1}"] = model

    img = rasterio.open(geotiff_path, tiled=True, blockxsize=256, blockysize=256)
    h, w = img.height, img.width
    stride = patch_size - 2 * border
    valid_size = stride
    offsets = get_patch_offsets(w, h, stride)

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

        # predict using retrained models
        if len(checkpoint_path) > 1:
            pred = ensemble_union_predict(models, patch)
        else:
            model = models['model_1']
            logits = model(patch)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).to(torch.uint8)

        # replace in the output mask
        pred = pred.squeeze().cpu().numpy()
        valid_h = min(valid_size, h - (row_off + border))
        valid_w = min(valid_size, w - (col_off + border))

        start_row = row_off + border
        start_col = col_off + border
        end_row = start_row + valid_h
        end_col = start_col + valid_w

        output_mask[
            start_row:end_row,
            start_col:end_col
        ] = pred[:valid_h, :valid_w]


    profile = img.profile.copy()
    profile.update(count=1,
                   dtype=rasterio.uint8,
                   compress='deflate',
                   tiled=True,
                   BIGTIFF='YES',
                   nodata=0
                   )
    img.close()

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output_mask, 1)


class ArcticPredict:
    def __init__(self, **kwargs):
        self.images = sorted(glob(os.path.join(kwargs['image_dir'], '*.tif')))
        self.pretrained_model = kwargs['pretrained_model']
        self.ensemble = kwargs['ensemble']
        self.set_name = kwargs['set_name']
        self.out_dir = kwargs['out_dir']
        self.patch_size = kwargs['patch_size']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):

        for image_path in self.images:
            print(f"Processing {image_path}")
            output_path = os.path.join(self.out_dir, os.path.basename(image_path))
            apply_model_on_geotiff(geotiff_path=image_path,
                                   checkpoint_path=self.pretrained_model,
                                   output_path=output_path,
                                   device=self.device,
                                   patch_size=self.patch_size)
            print(f'Finished processing and saved to {output_path}')


