import os
import math
import requests
import argparse
import wget
import csv

import shapely
from pathlib import Path
from tqdm.auto import tqdm

import geopandas as gpd
import pandas as pd
import numpy as np

import rasterio as rio
import rasterio.merge

from loguru import logger

def loader(args):

    account_name = 'elevationeuwest'
    container_name = 'copernicus-dem'
    account_url = 'https://' + account_name + '.blob.core.windows.net'
    blob_root = account_url + '/' + container_name + '/'

    cdem_content_extension = '.tif'

    cdem_90_extents_url = account_url + '/'
    cdem_90_extents_url += container_name + '/' + 'index/copernicus-dem-90-bounds.csv'
    os.makedirs(args.data_dir, exist_ok=True)

    fn = os.path.join(args.data_dir, cdem_90_extents_url.split('/')[-1])
    logger.info("loading cdem 90 extents url file")
    if not os.path.isfile(fn):
        wget.download(cdem_90_extents_url, fn, bar=None)

    # Create geopandas with blob data
    df = pd.read_csv(fn)
    geometry = [shapely.geometry.box(l, b, r, t)
                    for l, b, r, t in zip(df.left, df.bottom, df.right, df.top)]
    geodf = gpd.GeoDataFrame(df.blob_name, geometry=geometry, crs='epsg:4326')
    # Load grid cells info
    grid_cells =  gpd.read_file(args.grid_cells)
    l, b, r, t = grid_cells.unary_union.convex_hull.bounds
    area_data = shapely.geometry.box(l, b, r, t)

    # Download dem files:
    logger.info("Download dem files")
    data_root = Path(args.data_dir) / "COP90_hh"
    os.makedirs(data_root.as_posix(), exist_ok=True)

    filenames = []
    for blob_name in tqdm(geodf[
            ~geodf.intersection(area_data).geometry.is_empty].blob_name):
        fn = data_root / blob_name.split("/")[-1]
        filenames.append(fn.as_posix())
        if not os.path.isfile(fn):
            wget.download(blob_root + blob_name, fn.as_posix(), bar=None)

    # Merge files to one
    logger.info("Merge files to one")
    if len(filenames) == 0:
        logger.error("Not found files for downloading")
        return

    # Load singl file as source
    src = rio.open(filenames[0])
    out_meta = rio.open(filenames[0]).meta.copy()
    # Merge all files
    mosaic, out_trans = rio.merge.merge(
            [rio.open(filename) for filename in filenames]
        )

    # Update the metadata
    out_meta.update({"driver": "GTiff",
                      "height": mosaic.shape[1],
                      "width": mosaic.shape[2],
                      "transform": out_trans,
    #                   "crs": "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
                      }
                     )

    mosaic, out_trans = rio.merge.merge(
                [rio.open(filename) for filename in filenames]
            )

    out_fp = Path(args.data_dir) / args.output_file

    # Save file
    with rasterio.open(out_fp.as_posix(), "w", **out_meta) as dest:
        dest.write(mosaic)

    logger.info("Success")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
            default='data/copernicus_dem', help='Data dir')
    parser.add_argument('--grid_cells',
            default='development/grid_cells.geojson', help='Path to grid_cells.geojson')
    parser.add_argument('--output_file',
            default='COP90.tif', help='Name for the output file')
    args = parser.parse_args()

    loader(args)
