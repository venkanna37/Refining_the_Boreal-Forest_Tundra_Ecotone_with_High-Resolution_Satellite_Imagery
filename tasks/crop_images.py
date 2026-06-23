import os
import rasterio
from rasterio.windows import from_bounds
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from rasterio.warp import reproject, Resampling

season = 'summer'
big_tif_images = glob(f'/home/venky/Documents/projects/data/arctic/venky_predictions/bw15_1_majority_{season}/*.tif')
small_tif_images = glob('/home/venky/Documents/projects/arctic/data/test/images/*.tif')

small_tif_images = [image for image in small_tif_images if season in image]
output_directory = f'/home/venky/Documents/projects/data/arctic/test_pred_from_mosaic/{season}/'

for small_tif in small_tif_images:
    site_id = os.path.basename(small_tif)[:7]
    big_tif = [image for image in big_tif_images if site_id in image][0]

    with rasterio.open(big_tif) as src_big, rasterio.open(small_tif) as src_small:
        dst_array = np.zeros(
            (src_big.count, src_small.height, src_small.width),
            dtype=src_big.dtypes[0]
        )

        for band in range(1, src_big.count + 1):
            reproject(
                source=rasterio.band(src_big, band),
                destination=dst_array[band - 1],
                src_transform=src_big.transform,
                src_crs=src_big.crs,
                dst_transform=src_small.transform,
                dst_crs=src_small.crs,
                resampling=Resampling.nearest
            )

        profile = src_big.profile.copy()
        profile.update(
            height=src_small.height,
            width=src_small.width,
            transform=src_small.transform,
            crs=src_small.crs
        )

    # Save subset
    output_tif = os.path.join(output_directory, os.path.basename(small_tif))
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(dst_array)

    print("Subset saved:", output_tif)