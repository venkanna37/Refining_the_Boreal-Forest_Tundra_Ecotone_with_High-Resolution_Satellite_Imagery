import rasterio
import numpy as np
# geotiff_path = '/home/venky/Documents/projects/data/arctic/venky_predictions/image_bw15_new_majority/ArcticAOI6_6931_GE01-QB02-WV01-WV02-WV03_P_232_429_mosaic.tif'
geotiff_path = 'file:///home/venky/Documents/projects/data/arctic/venky_predictions/bw15_majority_latest/ArcticAOI6_6931_GE01-QB02-WV01-WV02-WV03_P_232_429_mosaic.tif'
with rasterio.open(geotiff_path) as src:
    patch_img = src.read([1])
    print(patch_img.sum())