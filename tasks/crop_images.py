import os
import rasterio
from rasterio.windows import from_bounds
from glob import glob
import matplotlib.pyplot as plt

season = 'spring'
big_tif_images = glob(f'/home/venky/Documents/projects/data/arctic/venky_predictions/bw15_1_majority_{season}/*.tif')
small_tif_images = glob('/home/venky/Documents/projects/arctic/data/test/images/*.tif')

small_tif_images = [image for image in small_tif_images if season in image]
output_directory = f'/home/venky/Documents/projects/data/arctic/test_pred_from_mosaic/{season}/'

for small_tif in small_tif_images:
    site_id = os.path.basename(small_tif)[:7]
    big_tif = [image for image in big_tif_images if site_id in image][0]

    # Open the reference (small) image
    with rasterio.open(small_tif) as small_src:
        org_image = small_src.read(1)
        bounds = small_src.bounds
        small_profile = small_src.profile

    # Open the large image
    with rasterio.open(big_tif) as big_src:

        # Create window corresponding to small image extent
        window = from_bounds(
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
            transform=big_src.transform
        )

        # Read data from large image
        data = big_src.read(window=window)

        # Update profile to match extracted data
        profile = big_src.profile.copy()
        profile.update(
            height=data.shape[1],
            width=data.shape[2],
            transform=big_src.window_transform(window)
        )

    # Save subset
    output_tif = os.path.join(output_directory, os.path.basename(small_tif))
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(data)

    print("Subset saved:", output_tif)