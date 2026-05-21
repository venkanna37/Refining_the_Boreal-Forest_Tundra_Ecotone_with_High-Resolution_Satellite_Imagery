import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import box

# Paths
sh_dir = "/home/venky/Documents/projects/data/arctic/new_area/new_location.shp"
raster_path = "/home/venky/Documents/projects/data/arctic/mosaics/ArcticAOI6_6931_GE01-QB02-WV01-WV02-WV03_P_232_429_mosaic.tif"
out_dir = "/home/venky/Documents/projects/arctic/data/new_location"

# Load vector data
gdf = gpd.read_file(sh_dir)
bbox_geoms = []
image_id = 0

# Open raster ONCE
with rasterio.open(raster_path) as src:

    # Reproject vector to raster CRS
    if gdf.crs != src.crs:
        gdf = gdf.to_crs(src.crs)

    # Iterate geometries
    for i, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        # Use bounding box of geometry
        minx, miny, maxx, maxy = geom.bounds
        bbox_geom = box(minx, miny, maxx, maxy)

        # Clip raster
        clipped_image, clipped_transform = mask(
            src,
            [bbox_geom],
            crop=True
        )

        bands, height, width = clipped_image.shape

        # Patch extraction (512x512)
        for j in range(0, height, 512):
            for k in range(0, width, 512):

                patch = clipped_image[:, j:j+512, k:k+512]
                # skip incomplete patches (optional)
                if patch.shape[1] != 512 or patch.shape[2] != 512:
                    continue

                window = Window(k, j, 512, 512)
                patch_transform = window_transform(window, clipped_transform)

                # Patch bounding box (correct geospatial footprint)
                left, bottom, right, top = rasterio.transform.array_bounds(
                    512, 512, patch_transform
                )
                patch_bbox = box(left, bottom, right, top)
                intersection = patch_bbox.intersection(geom).area
                union = patch_bbox.area
                iou = intersection/union
                print(f'IoU: {iou}')

                if iou < 0.8:
                    continue
               # Save image
                image_path = os.path.join(out_dir, 'images', f"summer_{image_id}_{j}_{k}.tif")
                meta = src.meta.copy()
                meta.update({
                    "driver": "GTiff",
                    "height": 512,
                    "width": 512,
                    "transform": patch_transform
                })
                with rasterio.open(image_path, "w", **meta) as dst:
                    dst.write(patch)

                # Save label (placeholder)
                label = np.zeros((1, 512, 512), dtype=np.uint8)
                label_path = os.path.join(out_dir, 'labels', f"summer_{image_id}_{j}_{k}.tif")
                with rasterio.open(label_path, "w", **meta) as dst:
                    dst.write(label)

                # Store bbox metadata
                bbox_geoms.append({
                    "image_id": image_id,
                    "geometry": patch_bbox
                })

        image_id += 1

# -----------------------------
# Save GeoDataFrame
# -----------------------------
gdf_bbox = gpd.GeoDataFrame(
    bbox_geoms,
    geometry="geometry",
    crs=gdf.crs
)

output_geojson = os.path.join(out_dir, "new_location_bbox.geojson")
gdf_bbox.to_file(output_geojson, driver="GeoJSON")

print("Done. Patches + GeoJSON saved.")
#
# sh_dir = '/home/venky/Documents/projects/data/arctic/new_area/new_location.shp'
# raster_path = '/home/venky/Documents/projects/data/arctic/mosaics/ArcticAOI6_6931_GE01-QB02-WV01-WV02-WV03_P_232_429_mosaic.tif'
# out_dir = '/home/venky/Documents/projects/arctic/data/new_location/'
#
# gdf = gpd.read_file(sh_dir)
# gdf.boundary.plot()
# plt.show()
# bbox_geoms = []
# image_id = 0
#
# for i, row in gdf.iterrows():
#     print(row['geometry'])
#     # Open raster
#     with rasterio.open(raster_path) as src:
#
#         assert gdf.crs != src.crs, 'check this'
#
#         # Convert geometries to GeoJSON-like format
#         # import ipdb;
#         # ipdb.set_trace()
#         minx, miny, maxx, maxy = row.geometry.bounds
#         geometries = box(minx, miny, maxx, maxy)
#
#         # Clip raster
#         clipped_image, clipped_transform = mask(src, [geometries], crop=True)
#
#         # shape of the image
#         _, h, w = clipped_image.shape
#
#         # clip clipped images to 512x512 patchs
#         for j in range(0, w, 512):
#             for k in range(0, h, 512):
#                 clipped_patch = clipped_image[:, j:j+512, k:k+512]
#                 # Save clipped patch
#                 output_path = os.path.join(out_dir, f"new_location_image_{image_id}_{j}_{k}.tif")
#                 label_path = os.path.join(out_dir, f"new_location_label_{image_id}_{j}_{k}.tif")
#                 clipped_meta = src.meta.copy()
#                 clipped_meta.update({
#                     "driver": "GTiff",
#                     "height": clipped_patch.shape[1],
#                     "width": clipped_patch.shape[2],
#                     "transform": rasterio.transform.from_origin(minx + k * src.res[0], maxy - j * src.res[1], src.res[0], src.res[1])
#                 })
#                 with rasterio.open(output_path, "w", **clipped_meta) as dest:
#                     dest.write(clipped_patch)
#
#                 labels = np.zeros_like(clipped_patch)
#                 with rasterio.open(label_path, "w", **clipped_meta) as dest:
#                     dest.write(labels)
#
#                 bbox_geoms.append({
#                     'image_id': image_id,
#                     'geometry': geometries
#                 })
#     image_id += 1
#
#
# # create dataframe using bbox_geoms and save as GeoJson
#
# gdf_bbox = gpd.GeoDataFrame(bbox_geoms, geometry='geometry')
# gdf_bbox.to_file(os.path.join(out_dir, os.path.join(out_dir, "new_location_bbox.geojson")))