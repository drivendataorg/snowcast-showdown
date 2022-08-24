import os
import time
import tempfile
import argparse
from glob import glob
from pathlib import Path
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import geopandas as gpd

import re
import requests

import xarray as xr
import rasterio as rio
import dask

import math
import torch
import torch.nn as nn
import torch.fft

from torch.nn.parameter import Parameter

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels,
            torch.div(x.size(-1), 2, rounding_mode='floor') + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class SpectralBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes,
                 kernel_size=1, stride=1, bias=False, activator=nn.ReLU):
        super(SpectralBlock1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        # Fourier domain
        self.spectr = SpectralConv1d(self.in_channels, self.out_channels,
                                            self.modes)
        # Feature domain
        self.linear = nn.Conv1d(self.in_channels, self.out_channels,
                                kernel_size=kernel_size, padding=kernel_size//2,
                                stride=stride, bias=bias)
        # Normalize
        self.bn = nn.BatchNorm1d(self.out_channels)
        self.activator = activator()

    def forward(self, x):
        # Features domain forward
        x2 = self.linear(x)
        # Fourier domain forward
        x1 = self.spectr(x)
        # Add time and feature
        x = self.bn(x1 + x2)
        x = self.activator(x)

        return x

class SnowNet(nn.Module):
    def __init__(self, features=6, h_dim=32,
                 width=48, timelag=21, out_dim=1, embedding_dim = 3):

        super(SnowNet, self).__init__()
        self.features = features
        self.timelag = timelag
        self.modes, self.width = width // 2, width
        self.h_dim, self.in_dim = h_dim, timelag
        self.k = 2
        self.embs = nn.Embedding(num_embeddings = 20,
                                 embedding_dim = embedding_dim)
        self.embs.weight.data[0] *= 0

        if timelag == width:
            self.fc0 = nn.Identity()
        else:
            self.fc0 = nn.Linear(self.timelag, self.width)

        self.step0 = nn.Sequential(
            nn.Conv1d(self.features, self.h_dim, 1, bias = False),
            nn.BatchNorm1d(self.h_dim), nn.PReLU(),
            nn.Conv1d(self.h_dim, self.h_dim, 1),
        )

        # Conv layers:
        self.conv0dem = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, bias=False),
            nn.BatchNorm2d(6), nn.PReLU(),
            nn.Conv2d(in_channels = 6, out_channels = 9, kernel_size = 3, bias = False),
            nn.BatchNorm2d(9), nn.PReLU(),
            nn.Conv2d(in_channels = 9, out_channels = 6, kernel_size = 3, bias = False),
            nn.Flatten(),
            nn.Linear(6*4*4, self.width),
        )

        self.conv1soil = nn.Sequential(
            nn.Conv2d(embedding_dim, 6, kernel_size=3, bias=False),
            nn.BatchNorm2d(6), nn.PReLU(),
            nn.Conv2d(in_channels = 6, out_channels = 9, kernel_size = 3, bias = False),
            nn.BatchNorm2d(9), nn.PReLU(),
            nn.Conv2d(in_channels = 9, out_channels = 6, kernel_size = 3, bias = False),
            nn.Flatten(),
            nn.Linear(6*4*4, self.width),
        )

        self.layers = nn.Sequential(
            SpectralBlock1d(self.h_dim + 2, self.h_dim, int(self.modes*1/2),
                            activator=nn.PReLU),
            SpectralBlock1d(self.h_dim, self.h_dim, int(self.modes*1/2),
                            activator=nn.PReLU),
            SpectralBlock1d(self.h_dim, self.h_dim, int(self.modes*1/2),
                            activator=nn.PReLU),
            SpectralBlock1d(self.h_dim, self.h_dim, int(self.modes*1/4),
                            activator=nn.PReLU),
        )

        self.step_t = nn.Conv1d(self.h_dim, 1, 1)
        self.fc1 = nn.Linear(width, out_dim)

    def forward(self, x, dem, soil):

        x = self.fc0(x)
        x = self.step0(x)

        soil = self.embs(soil.long()).permute(0,3,1,2)
        d0 = self.conv0dem(dem)
        d1 = self.conv1soil(soil)
        x = torch.cat([
            x, d0.view(-1, 1, self.width), d1.view(-1, 1, self.width)], 1)

        x = self.layers(x)

        x = self.step_t(x)
        x = self.fc1(x)

        return x.squeeze(-1)

class ModelAggregator(nn.Module):
    def __init__(self, models):
        super(ModelAggregator, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, *args):
        x_a = torch.stack([
            m(*args) for m in self.models], dim=-1)
        return x_a.clamp(0).mean(-1)

# Modis projection
PROJ4MODIS = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
# HRRR projection
PROJ4HRRR = '+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs'

def load_sundecline(date_range):

    grid_cells = gpd.read_file(args.grid_cells)

    # Calculate additional values for sunlingt duration
    grid_cells['lat'] = (grid_cells.geometry.bounds['maxy'] + grid_cells.geometry.bounds['miny']) / 2
    grid_cells['lon'] = (grid_cells.geometry.bounds['maxx'] + grid_cells.geometry.bounds['minx']) / 2
    grid_cells['lat_rad'] = np.pi * grid_cells['lat'] / 180
    grid_cells['tan_lat'] = np.tan(grid_cells['lat_rad'])
    grid_cells['k_cos'] = np.cos(np.pi * 90.833 / 180) / np.cos(grid_cells['lat_rad'])

    # Load cvs file with sun decline information
    sun_decline = pd.read_csv(args.sun_decline, index_col=[0], parse_dates=[0])
    # Caclculate values
    time_idx = date_range

    sun_duration = grid_cells.loc[:, "k_cos"].values[None] * sun_decline.loc[time_idx, "cos-1_decl"].values[:, None]
    sun_duration -= grid_cells.loc[:, "tan_lat"].values[None] * sun_decline.loc[time_idx, "tan_decl"].values[:, None]
    sun_duration = 8 * 180 * np.arccos(sun_duration) / (np.pi * 1)
    sun_duration = sun_duration.astype(np.float32)
    attrs = {
        'long_name': "Sunlight Duration",
        'shortName': "sd",
        'units': "minutes per day",
        'reference': "https://gml.noaa.gov/grad/solcalc/calcdetails.html"
    }

    return sun_duration, attrs

def load_hrrr(date_range):

    grid_cells = gpd.read_file(args.grid_cells)
    # Obtaine x/y projection grid from sample with rasterio:
    ds = xr.open_dataset(args.hrrr_sample, engine='rasterio')

    proj_y = np.flip(ds.y)
    proj_x = ds.x
    del ds

    # Search points: HRRR projection
    mid_x = grid_cells.to_crs(PROJ4HRRR).geometry.centroid.x.values
    mid_y = grid_cells.to_crs(PROJ4HRRR).geometry.centroid.y.values

    mid_x = xr.DataArray(mid_x, dims="cell_id")
    mid_y = xr.DataArray(mid_y, dims="cell_id")

    tmp_dir = tempfile.TemporaryDirectory(dir="/data")
    num_chunks = np.ceil(len(date_range) / 7).astype(int)

    for i, dates in enumerate(np.array_split(date_range, num_chunks)):
        ds = get_points(dates, proj_x = proj_x, proj_y = proj_y,
                        mid_x = mid_x, mid_y = mid_y, hrrr_dir=Path(args.hrrr_dir))
        # Save to file:
        ds.to_netcdf(
            f"{tmp_dir.name}/hrrr_{i}.nc", format="NETCDF4", engine='netcdf4')
        ds.close();
        del ds

    hrrr_ds = xr.open_mfdataset(f"{tmp_dir.name}/hrrr_*.nc", engine='netcdf4')
    hrrr_ds.load()
    hrrr_ds.close()
    tmp_dir.cleanup()

    return hrrr_ds

def load_modis(date_range):

    grid_cells = gpd.read_file(args.grid_cells)

    # Load modis:
    bounds = grid_cells.to_crs(PROJ4MODIS).geometry.bounds
    # Data slice size
    rx, ry = 5, 3
    # Transform values
    a, _, b, _, c, d = 463.31271652791725, 0.0, -11119505.196667, 0.0, -463.31271652750013, 5559752.598333
    rowsn = (bounds.maxy.values - d ) / c
    colsn = (bounds.minx.values - b ) / a
    xs = xr.DataArray(
        np.tile( np.stack(
            [np.arange(x, x + rx) for x in np.floor(colsn).astype(int)]), (1,1,ry)).flatten())
    ys = xr.DataArray(
        np.repeat( np.stack(
            [np.arange(x, x + ry) for x in np.floor(rowsn).astype(int)]), rx, axis=-1).flatten())

    modis_ds =  xr.concat([get_data(day,
             xs, ys, rx=5, ry=3,
             cell_id = grid_cells.cell_id.values,
             product = 'MYD10A1', variable = 'NDSI',
             data_dir = args.modis_dir) for day in date_range],
        dim='time').transpose("time", "cell_id", "x", "y")

    return modis_ds.compute()

def round_time(ds):
    ds.coords['time'] = ds.coords['time'].dt.floor('D')
    return ds

def get_points(date_range, proj_x, proj_y, mid_x, mid_y, hrrr_dir):
    fnamest12 = []
    for day in date_range:
        for cycle in [12,11,10]:
            if f"{day:%Y%m%d}" == "20160805": cycle = 10
            filename = hrrr_dir / f"hrrr.{day:%Y%m%d}/conus/hrrr.t{cycle:02}z.wrfsfcf00.grib2"
            if filename.is_file():
                fnamest12.append(filename.as_posix())
                break
    fnamest00 = []
    for day in date_range:
        for cycle in [0,1,2]:
            filename = hrrr_dir / f"hrrr.{day:%Y%m%d}/conus/hrrr.t{cycle:02}z.wrfsfcf00.grib2"
            if filename.is_file():
                fnamest00.append(filename.as_posix())
                break
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        ds = xr.merge([
        # Temperature T12
        xr.open_mfdataset(fnamest12, engine='cfgrib',
                             backend_kwargs={'indexpath':''},
                             drop_variables = ['latitude', 'longitude', 'valid_time', 'step'],
                             filter_by_keys={'stepType': 'instant',
                             'typeOfLevel': 'surface', 'shortName': 't'},
                             preprocess = round_time,
            concat_dim='time', combine='nested', parallel=False).rename({'t': 't12'}),
        # U component of wind
        xr.open_mfdataset(fnamest12, engine='cfgrib',
                             backend_kwargs={'indexpath':''},
                             drop_variables = ['latitude', 'longitude', 'valid_time', 'step'],
                             filter_by_keys={
                             'stepType': 'instant',
                             'typeOfLevel': 'heightAboveGround',
                             'shortName': 'u'},
                             preprocess = round_time,
                             concat_dim='time', combine='nested', parallel=False),
        # V component of wind
        xr.open_mfdataset(fnamest12, engine='cfgrib',
                             backend_kwargs={'indexpath':''},
                             drop_variables = ['latitude', 'longitude', 'valid_time', 'step'],
                             filter_by_keys={
                             'stepType': 'instant',
                             'typeOfLevel': 'heightAboveGround',
                             'shortName': 'v'},
                             preprocess = round_time,
                             concat_dim='time', combine='nested', parallel=False),
        # Water equivalent of accumulated snow depth
        xr.open_mfdataset(fnamest12, engine='cfgrib',
                             backend_kwargs={'indexpath':''},
                             drop_variables = ['latitude', 'longitude', 'valid_time', 'step'],
                             filter_by_keys={'stepType': 'instant',
                             'typeOfLevel': 'surface', 'shortName': 'sdwe'},
                             preprocess = round_time,
                             concat_dim='time', combine='nested', parallel=False),
        # Precipitable water
        xr.open_mfdataset(fnamest12, engine='cfgrib',
                             backend_kwargs={'indexpath':''},
                             drop_variables = ['latitude', 'longitude', 'valid_time', 'step'],
                             filter_by_keys={'stepType': 'instant',
                                'typeOfLevel': 'atmosphereSingleLayer',
                                'shortName': 'pwat'},
                             preprocess = round_time,
                             concat_dim='time', combine='nested', parallel=False),
        # Maximum/Composite radar reflectivity
        xr.open_mfdataset(fnamest12, engine='cfgrib',
                             backend_kwargs={'indexpath':''},
                             drop_variables = ['latitude', 'longitude', 'valid_time', 'step'],
                             filter_by_keys={'stepType': 'instant',
                                'typeOfLevel': 'atmosphere',
                                'shortName': 'refc'},
                             preprocess = round_time,
                             concat_dim='time', combine='nested', parallel=False),
        # Temperature T00
        xr.open_mfdataset(fnamest00, engine='cfgrib',
                             backend_kwargs={'indexpath':''},
                             drop_variables = ['latitude', 'longitude', 'valid_time', 'step'],
                             filter_by_keys={'stepType': 'instant',
                             'typeOfLevel': 'surface', 'shortName': 't'},
                             preprocess = round_time,
            concat_dim='time', combine='nested', parallel=False).rename({'t': 't00'}),

        # Water equivalent of accumulated snow depth - Day accumulated
        xr.open_mfdataset(fnamest00, engine='cfgrib',
                             backend_kwargs={'indexpath':''},
                             drop_variables = ['latitude', 'longitude', 'valid_time', 'step'],
                             filter_by_keys={
                             'stepType': 'accum',
                             'typeOfLevel': 'surface',
                             'shortName': 'sdwe'},
                             preprocess = round_time,
            concat_dim='time', combine='nested', parallel=False).rename({'sdwe': 'sdwea'})
    ]).reindex({'time': date_range})

    ds['x'] = proj_x
    ds['y'] = proj_y

    points = ds.sel(x=mid_x, y=mid_y, method="nearest")
    del ds
    return points.compute()

def flatten(outter):
    return [item for sublist in outter for item in sublist]

def get_data(day, xs, ys, rx, ry, cell_id, product, variable, data_dir):
    # filenames for reading
    filenames = flatten([
                glob(f"{data_dir}/{product}/{h:0>2d}/{v:0>2d}/{day:%Y%j}/{product}.A{day:%Y%j}.*.hdf")
                     for h, v in [(8,4),(8,5),(9,4),(9,5),(10,4),(10,5)]])
    if len(filenames) > 4:
        xds = xr.open_mfdataset(filenames, engine='rasterio', variable=variable)
        ds = xr.Dataset(
            data_vars = {
                variable : (
                    ["cell_id", "time", "x", "y"],
                            xds[variable].isel(x=xs, y=ys).data.reshape(-1, 1, ry, rx))
            },
            coords = dict(
                    cell_id = cell_id,
                    time = pd.date_range(day, day)
                ),
        )
    else:
        # No files for reading
        ar = np.empty((cell_id.shape[0], 1, ry, rx), dtype=np.float32)
        ar.fill(np.nan)
        ds = xr.Dataset(
            data_vars = {
                variable : (["cell_id", "time", "x", "y"], ar)
            },
            coords = dict(
                    cell_id = cell_id,
                    time = pd.date_range(day, day)
                ),
        )
    return ds.compute()

def update_dataset(ds, date_range):

    ndate_range = date_range[~np.isin(date_range, ds.time.data)]
    if len(ndate_range) == 0:
        return ds
    print(ndate_range)
    # Extract data
    print("Start update HRRR")
    hrrr_ds = load_hrrr(ndate_range)
    print("Start update MODIS")
    modis_ds = load_modis(ndate_range)
    # Merge datasets
    nds = xr.merge([hrrr_ds, modis_ds])
    sun_duration, attrs = load_sundecline(ndate_range)
    nds = nds.assign(dict(
        sd = (['time', 'cell_id'], sun_duration, attrs), ))

    return xr.merge([ds, nds])

def run_estimator(args):
    day = args.date - timedelta(days=1)
    start_date = day.replace(hour=0, minute=0, second=0, microsecond=0)
    print(start_date)
    if start_date.weekday() != 3:
        print("Prediction only for Thursdays")
        return 0
    grid_cells = gpd.read_file(args.grid_cells)
    ds = xr.open_dataset(args.dataset_file, engine='netcdf4', decode_times=True)
    ds.load();
#     ds = ds.loc[{"time" : slice(None,"2022-06-07")}]
    ds.close();

    date_range = pd.date_range(
        start_date - timedelta(92), start_date,
         closed='left', freq='1D').tz_localize(None)

    # Cheack dates
    print("Start update")
    ds = update_dataset(ds, date_range)
    print("Updatetd")
    # Save to file:
    ds.to_netcdf(f"{args.dataset_file}", format="NETCDF4", engine='netcdf4')
    ds.close();

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

    band_values = np.array(band.ffill('time').fillna(0).transpose(
        "cell_id", "feature", "time").data)

    images_dem = ds.dem.data
    images_soil = ds.soil.data

    subm = pd.read_csv(args.format_file, index_col=[0])

    ## submission
    models = []
    for fold_idx in range(5):
        model = SnowNet(features=10, h_dim=64, width=92, timelag=92)
        model.load_state_dict(
            torch.load(f'{args.model_dir}/SnowNet_fold_{fold_idx}_last.pt')['model']
        )
        models.append(model)
    model = ModelAggregator(models)
    model.eval();

    features = torch.from_numpy(band_values).float()
    dem = torch.from_numpy(images_dem / 1000 - 2.25).float().unsqueeze(1)
    soil = torch.from_numpy(images_soil).long()

    with torch.no_grad():
        result = model(features, dem, soil).clamp(0)
        result = result.detach().cpu().squeeze().numpy()

    subm.loc[ds.cell_id.data, f"{start_date:%Y-%m-%d}"] = result
    subm.to_csv(args.format_file)
    print(subm.head())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', "--date",
            default="2022-02-17T20:55:08.914461+00:00", help='Prediction date')

    args = parser.parse_args()
    args.date = datetime.fromisoformat(args.date)
    args.timelag = 92
    args.hrrr_dir ='/data/hrrr'
    args.modis_dir = '/data/modis'
    args.grid_cells ='/data/grid_cells.geojson'
    args.dem_file = '/data/copernicus_dem/COP90_hh.tif'
    args.soil_file = '/data/global_soil_regions/so2015v2.tif'
    args.hrrr_sample = '/data/hrrr_sample.grib2'
    args.sun_decline = '/data/sun_decline.csv'
    args.model_dir = '/data/weights'
    args.dataset_file = '/data/evaluation_dataset.nc'
    args.format_file ='/data/submission_format.csv'
    args.output_file ='/data/submission_format.csv'

    run_estimator(args)
