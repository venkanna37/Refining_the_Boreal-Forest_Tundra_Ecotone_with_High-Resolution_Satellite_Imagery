# Edited by: Connor Anderson (SSAI/NASA GSFC)
# Author: Jesse Meyer (SSAI/NASA GSFC)

from osgeo import gdal, ogr
from numpy import zeros, uint8, array, float32, radians, cos, sin, sqrt
from numpy.linalg import norm
import os
from os import environ, makedirs, system
from os.path import isdir, isfile
from shutil import copy2, copyfile, rmtree
from time import time
import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert tree predictions to polygons.',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--input', type=os.path.abspath, metavar='PATH', required = True, 
                        help="""Path to binary NN classification geopackage.""")
    parser.add_argument('-m', '--landMask', type=os.path.abspath, metavar='PATH', required = True, 
                        help="""Path to land cover classification. Designed to run with Zhang et al. 2024 GLC_FCS30D.""")
    args = parser.parse_args()


    in_gpkg_fps = [args.input]


    start = time() / 60

    gdal.UseExceptions()
    ogr.UseExceptions()

    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
    gdal.SetConfigOption("NUM_THREADS", "ALL_CPUS")
    gdal.SetConfigOption("OGR_SQLITE_CACHE", "128")

    disk_create_options = ['COMPRESS=DEFLATE', 'PREDICTOR=2', 'INTERLEAVE=BAND', 'Tiled=YES', 'NUM_THREADS=ALL_CPUS', 'SPARSE_OK=True']

    land_mask_fp = args.landMask

    assert isfile(land_mask_fp), land_mask_fp

    lm_ds = gdal.Open(land_mask_fp)
    lm_gt = lm_ds.GetGeoTransform()

    lm_arr = None
    p_arr = None

    DENSITY_METERS_PER_PIXEL = 100
    coverage_arr = zeros((160, 160), float32) #100m density from 32000 half meter raster

    for in_gpkg_fp in in_gpkg_fps:
        print(in_gpkg_fp)
        assert isfile(in_gpkg_fp), in_gpkg_fp

        gpkgDir = os.path.split(in_gpkg_fp)[0]
        predictions_fp = os.path.join(gpkgDir, '{}_NN_classification.tif'.format('_'.join(os.path.split(in_gpkg_fp)[1].split('_')[:-1])))


        p_ds = gdal.Open(predictions_fp)
        p_mem_ds = gdal.GetDriverByName("MEM").CreateCopy("", p_ds)
        p_ds = None
        if p_arr is None:
            p_arr = zeros((p_mem_ds.RasterYSize, p_mem_ds.RasterXSize), uint8)
            lm_arr = zeros((p_mem_ds.RasterYSize, p_mem_ds.RasterXSize), uint8)

        assert p_mem_ds.RasterYSize == lm_arr.shape[0], p_mem_ds.RasterYSize
        assert p_mem_ds.RasterXSize == lm_arr.shape[1], p_mem_ds.RasterXSize

        p_gt = p_mem_ds.GetGeoTransform()

        density_size = (p_mem_ds.RasterYSize * p_gt[5] / -DENSITY_METERS_PER_PIXEL, p_mem_ds.RasterXSize * p_gt[1] / DENSITY_METERS_PER_PIXEL)
        assert density_size == (160, 160), density_size

        lm_x_offset = int((p_gt[0] - lm_gt[0]) / lm_gt[1])
        lm_y_offset = int((p_gt[3] - lm_gt[3]) / lm_gt[5])
        lm_ds.ReadAsArray(lm_x_offset, lm_y_offset, p_mem_ds.RasterXSize * p_gt[1] / int(lm_gt[1]), p_mem_ds.RasterYSize * p_gt[5] / int(lm_gt[5]), buf_obj=lm_arr)

        ogr_dsk_ds = ogr.Open(in_gpkg_fp)
        vector_mem_ds = gdal.GetDriverByName("Memory").Create('', 0, 0, 0, gdal.GDT_Unknown) #NOTE(Jesse): GDAL has a highly unintuitive API
        vector_mem_ds.CopyLayer(ogr_dsk_ds.GetLayer(0), "trees")
        ogr_dsk_ds = None

        for i in range(vector_mem_ds.GetLayerCount()):
            layer = vector_mem_ds.GetLayerByIndex(i)
            print(layer.GetName())

        ro = gdal.RasterizeOptions(bands=[1], burnValues=0, SQLStatement="select GEOMETRY from trees t where ST_Area(ST_Envelope(GEOMETRY)) <= 2.001", SQLDialect="SQLITE", allTouched=False)
        gdal.Rasterize(p_mem_ds, vector_mem_ds, options=ro)
        vector_mem_ds = None

        p_mem_ds.ReadAsArray(buf_obj=p_arr)
        
        # NOTE(Connor): Desigend to run with Zhang et al (2024) land cover classification (https://doi.org/10.5281/ZENODO.8239304). Class value will need to change if using different product.
        WATER_BODIES = 210
        IMPERVIOUS = 190
        p_arr[(lm_arr == WATER_BODIES) | (lm_arr == IMPERVIOUS) | (p_arr <= 0.5)] = 0
        p_arr = p_arr*100

        raster_disk_ds = gdal.GetDriverByName("GTiff").Create(gpkgDir + "/NN_classification_lc_glc_filter.tif", xsize=p_mem_ds.RasterXSize, ysize=p_mem_ds.RasterYSize, bands=1, eType=gdal.GDT_Byte, options=disk_create_options)
        # raster_disk_ds.GetRasterBand(1).SetNoDataValue(p_mem_ds.GetRasterBand(1).GetNoDataValue())
        raster_disk_ds.GetRasterBand(1).SetNoDataValue(255)
        raster_disk_ds.SetGeoTransform(p_gt)
        raster_disk_ds.SetProjection(p_mem_ds.GetProjection())
        raster_disk_ds.GetRasterBand(1).WriteArray(p_arr)
        raster_disk_ds = None

        gdal.Warp(gpkgDir + "/NN_classification_lc_glc_filter_coverage.tif",
         gpkgDir + "/NN_classification_lc_glc_filter.tif",
         xRes=100.0,
         yRes=100.0,
         creationOptions=disk_create_options,
         resampleAlg=gdal.GRIORA_Average)

    p_arr = None
    coverage_arr = None
    lm_arr = None

    lm_ds = None

    stop = time() / 60
    print(f"Total time took {stop - start}s")

if __name__ == "__main__":
    main()