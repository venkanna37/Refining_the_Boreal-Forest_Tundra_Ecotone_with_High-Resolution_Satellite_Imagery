# Author: Connor Anderson (SSAI/NASA GSFC)

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import logging
import argparse
import time
from joblib import Parallel, delayed, cpu_count

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

def getDistances(seed, df, idx):
    '''Compute distances between seed polygon and target polygon
    :param seed: geopandas GeoDataFrame containing polygon to serve as base of distance calculation
    :param df: geopandas GeoDataFrame containing all target polygons
    :param idx: int, index of target polygon within df
    return: distance between seed and target polygon
    '''
    return [float(seed['geometry'].distance(df.loc[i]['geometry']).iloc[0]) for i in idx]

def buffer_and_join(orig_df, step, buffDist = 1000, new_df = None, runInParallel = False):
    '''Select polygons based on distance from neighboring polygons. Starts with largest polygon by area.
    :param orig_df: geopandas GeoDataFrame containing polygons
    :param step: int, denotes first selection or other
    :param buffDist: buffer distance with units according to crs, default is 1000 m
    :param new_df: geopandas GeoDataFrame, used for buffer after first selection iteration
    :param runInParallel: flag indicating whether processing should be done in parallel
    :return joinedPolys: geopandas GeoDataFrame, polygons from orig_df that intersect the buffer
    :return reDF: geopandas GeoDataFrame, polygons that were not within buffDist to the target, used in next iteration
    :return numJoined: int, number of polygons selected
    '''
    if step == 1:
        # Set seed polygon by selecting polygon with largest area
        logging.info('Setting Initial Seed Polygon')
        orig_df['Area'] = orig_df.area
        orig_df = orig_df.sort_values(by = 'Area', ascending = False).reset_index(drop = True)
        seed = orig_df[:3].dissolve()
        wDF = orig_df[3:].reset_index(drop=True)
        indexes = np.array_split(wDF.index.values, np.ceil(len(wDF.index.values)/1000))

        # Compute distances
        logging.info('Calculating distances')
        if runInParallel == True:
            allDistances = Parallel(n_jobs = 12, verbose = 1)(delayed(getDistances)(seed = seed, df = wDF, idx = i) for i in indexes)
        else:
            allDistances = [getDistances(seed = seed, df = wDF, idx = i) for i in indexes]
        wDF['SeedDistance'] = np.concatenate(allDistances)

        # Select polygons to keep
        intPolys = wDF[wDF.SeedDistance <= buffDist].reset_index(drop = True)
        reDF = wDF[wDF.SeedDistance > buffDist].reset_index(drop = True)
        numJoined = len(intPolys)
        logging.info('Creating new seed polygon')
        joinedPolys = pd.concat([seed, intPolys])[['geometry']].dissolve()

    else:
        seed, wDF = orig_df, new_df
        indexes = np.array_split(wDF.index.values, np.ceil(len(wDF.index.values)/1000))
        logging.info('Calculating distances')

        if runInParallel is True:
            allDistances = Parallel(n_jobs = 12, verbose = 1)(delayed(getDistances)(seed = seed, df = wDF, idx = i) for i in indexes)
        else:
            allDistances = [getDistances(seed = seed, df = wDF, idx = i) for i in indexes]
        wDF['SeedDistance'] = np.concatenate(allDistances)

        # Select polygons to keep
        intPolys = wDF[wDF.SeedDistance <= buffDist].reset_index(drop = True)
        reDF = wDF[wDF.SeedDistance > buffDist].reset_index(drop = True)
        numJoined = len(intPolys)
        logging.info('Creating new seed polygon')
        joinedPolys = pd.concat([seed, intPolys])[['geometry']].dissolve()

    return joinedPolys, reDF, numJoined

def main():
    parser = argparse.ArgumentParser(description='Iteratively identify polygons within buffer distance.', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--inputGPKG', type=os.path.abspath, metavar='PATH', required = True, 
                        help="""Path to input geopackage.""")
    parser.add_argument('-o', '--outputFile', type=os.path.abspath, metavar='PATH', required = True,
                        help="""Path to output geopackage.""")
    parser.add_argument('-b', '--buffer', type=int, metavar='int', required = True,
                        help="""Buffer distance in meters """)
    parser.add_argument('-parallel', '--parallel', action='store_true',
                        help="""Flag to indicate parallel processing with Python's Parallel module. Default is no parallel processing.""")
    args = parser.parse_args()

    logging.info('Input Geopackage: {}'.format(args.inputGPKG))
    logging.info('Buffer Distance: {}'.format(args.buffer))
    runInParallel = False
    
    if args.parallel:
        from joblib import Parallel, delayed, cpu_count
        logging.info('Running using Parallel function in joblib.')
        runInParallel = True


    df = gpd.read_file(args.inputGPKG)

    # Loop through selection/buffer/overlay until no more polygons are identified.
    previousMerge = 1
    numMerge = 1
    n = 1
    step = 1
    while previousMerge != 0:
        tStart = time.time()
        previousMerge = n
        if step == 1:
            selectedPolys, newDF, numMerge = buffer_and_join(df, step = step, buffDist = args.buffer, runInParallel = runInParallel)
            logging.info('Polygon Count: {}'.format(numMerge))
        else:
            selectedPolys, newDF, numMerge = buffer_and_join(selectedPolys, new_df = newDF, step = step, buffDist = args.buffer, runInParallel = runInParallel)
            logging.info('Polygon Count: {}'.format(numMerge))
        logging.info('Step {} took {} minutes'.format(step, np.round(time.time()-tStart)/60, 3))
        n = numMerge
        step += 1
        
    selectedPolys.to_file(args.outputFile)

if __name__ == "__main__":
    main()
