"""
Code for evaluating on test data
Implemented for only UNet model
"""

import os
import torch
import rasterio
import argparse
import numpy as np
from tqdm import tqdm
from pipelines import network


@torch.no_grad()
def ensemble_union_predict(models, input_patch, mode = "majority"):
    """
    Args:
        models (list or dict): list/dict of PyTorch models (eval mode)
        input_patch (torch.Tensor): (B, C, H, W)

    Returns:
        union_mask (torch.Tensor): (B, 1, H, W) binary mask
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
    if mode == "majority":
        mask = torch.mean(stacked, dim=0)
        mask = (mask > 0.5).to(torch.uint8)
    elif mode == "intersection":
        # INTERSECTION
        mask = stacked > 0.5
        mask = torch.all(mask, dim=0).to(torch.uint8)

    return mask


def apply_model_on_geotiff(
        geotiff_path,
        checkpoint_path,
        output_path,
        device=None,
        rescale_value = 2000
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}

    for i, path in enumerate(checkpoint_path):
        model = network.TinyUNet()
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()
        models[f"model_{i + 1}"] = model


    with rasterio.open(geotiff_path) as src:
        image = src.read(1).astype(np.float32) / rescale_value
        image = np.expand_dims(image, 0)
        profile = src.profile

    image = torch.from_numpy(image)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    _, c, h, w = image.shape

    with torch.no_grad():
        patch = image.to(device)
        pred = ensemble_union_predict(models, patch)[0, 0]

    output_mask = pred.cpu().numpy()
    profile.update(count=1, dtype=rasterio.uint8)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output_mask.astype(np.uint8), 1)


def main():
    parser = argparse.ArgumentParser(description="Batch inference on GeoTIFF directory")
    parser.add_argument("--keywords", type=str, help="Keyword used to save the model")
    parser.add_argument("--input_dir", type=str, help="Directory with input GeoTIFFs",
                        default="data/data/testdata/images")
    parser.add_argument("--output_dir", type=str, help="Directory with input GeoTIFFs",
                        default="data/predictions/")
    parser.add_argument("--patch_size", type=int, default=512)

    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, f'pred_{keywords[0]}')
    checkpoint = [f"./runs/{i}/best_precision.pth" for i in keywords]

    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    tif_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(".tif")]

    for fname in tqdm(tif_files):
        in_path = os.path.join(args.input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        apply_model_on_geotiff(
            geotiff_path=in_path,
            checkpoint_path = checkpoint,
            output_path=out_path,
            patch_size=args.patch_size,
            device=device
        )


if __name__ == "__main__":
    main()