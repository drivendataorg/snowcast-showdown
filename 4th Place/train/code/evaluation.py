import pandas as pd
import numpy as np
import geopandas as gpd

import time
import os

from pathlib import Path
from datetime import datetime, date, timedelta

import xarray as xr

import requests
import argparse
import tempfile
import tqdm.auto as tq

from loguru import logger

from layers import *
from dataset import *

# Modis projection
PROJ4MODIS = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
# HRRR projection
PROJ4HRRR = '+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs'

def create_ds(date_range, args):

    # Extract data
    logger.info("Loading HRRR")
    hrrr_ds = load_hrrr(date_range, args)

    logger.info("Loading MODIS")
    modis_ds = load_modis(date_range, args)

    # Merge
    ds = xr.merge([hrrr_ds, modis_ds])

    logger.info("Loading Sun Duration data")
    sun_duration, attrs = load_sundecline(date_range, args)

    logger.info("Loading Static data")
    images_dem, images_soil = load_static(args)

    ds = ds.assign(dict(
        sd = (['time', 'cell_id'], sun_duration, attrs),
        dem = (["cell_id", "xlat", "ylon"], images_dem),
        soil = (["cell_id", "xlat", "ylon"], images_soil),
    ))
    return ds

def main(args):
    logger.info("Start loading data")

    grid_cells = gpd.read_file(args.grid_cells)

    # Download files if required:
    start_date = datetime.strptime(args.date, '%Y-%m-%d')
    date_range = pd.date_range(
            start_date - timedelta(args.timelag), start_date, closed='left', freq='1D')

    download_hrrr(date_range, args)
    download_modis(date_range, args)

    ds = create_ds(date_range, args)

    ds = ds.loc[{"time" : date_range}]

    band = xr.concat([
            (ds.t00 - 273.15) / 20,
            (ds.t12 - 273.15) / 20,
            (ds.sdwe**0.25 - 1),
            (ds.pwat - 8) / 7,
            ds.refc / 10,
            ds.u / 20,
            ds.v / 20,
            ds.sdwea,
            ds.NDSI.ffill('time').fillna(0).reduce(np.nanmean, ("x", "y")),
            (ds.sd / 200) - 3.6,
        ], dim = 'feature'
    )
    
    grid_cells = gpd.read_file(args.grid_cells)

    band_values = np.array(band.ffill('time').fillna(0).transpose(
        "cell_id", "feature", "time").data)

    images_dem = ds.dem.data
    images_soil = ds.soil.data

    logger.info("Loading model")
    models = []
    for fold_idx in range(5):
        model = SnowNet(features=10, h_dim=64, width=92, timelag=92)
        model.load_state_dict(
            torch.load(f'{args.model_dir}/SnowNet_fold_{fold_idx}_last.pt')['model']
        )
        models.append(model)
    model = ModelAggregator(models)
    model.eval();
    logger.info("Evaluating...")

    features = torch.from_numpy(band_values).float()
    dem = torch.from_numpy(images_dem / 1000 - 2.25).float().unsqueeze(1)
    soil = torch.from_numpy(images_soil).long()

    with torch.no_grad():
        result = model(features, dem, soil).clamp(0)
        result = result.detach().cpu().numpy()
    subm = pd.DataFrame(result,
                index=grid_cells.cell_id.values, columns=[args.date])
    subm.to_csv(args.output_file)

    logger.info("Evaluation completed ")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default='2022-02-03', help='Evaluation day')
    parser.add_argument('--hrrr_dir', default='data/hrrr', help='HRRR dir')
    parser.add_argument('--modis_dir', default='data/modis', help='MODIS dir')
    parser.add_argument('--grid_cells',
        default='evaluation/grid_cells.geojson', help='Path to grid_cells.geojson')
    parser.add_argument('--dem_file',
        default='data/copernicus_dem/COP90_hh.tif', help='Path to dem tif file')
    parser.add_argument('--soil_file',
        default='data/global_soil_regions/so2015v2.tif', help='Path to soil tif file')
    parser.add_argument('--hrrr_sample',
        default='hrrr_sample.grib2', help='HRRR single variable file')
    parser.add_argument('--sun_decline',
        default='sun_decline.csv', help='Sun decline file')
    parser.add_argument('--model_dir',
        default='weights', help='Model weights directory')
    parser.add_argument('--output_file',
            default='submission.csv', help='Name for the output file')
    args = parser.parse_args()

    main(args)
