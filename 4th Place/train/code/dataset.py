import os
from glob import glob
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
import gc
from datetime import datetime, date, timedelta
import tempfile
import tqdm.auto as tq
import wget
import requests

import xarray as xr
import rasterio as rio

from azure.storage.blob import ContainerClient

# Modis projection
PROJ4MODIS = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
# HRRR projection
PROJ4HRRR = '+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs'

def load_sundecline(date_range, args):

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

def list_blobs_in_folder(container_name, folder_name, modis_container_client):
    """
    List all blobs in a virtual folder in an Azure blob container
    """

    files = []
    generator = modis_container_client.list_blobs(name_starts_with=folder_name)
    for blob in generator:
        files.append(blob.name)
    return files


def list_hdf_blobs_in_folder(container_name, folder_name, modis_container_client):
    """"
    List .hdf files in a folder
    """

    files = list_blobs_in_folder(container_name, folder_name, modis_container_client)
    files = [fn for fn in files if fn.endswith('.hdf')]
    return files

def download_modis(date_range, args):

    modis_account_name = 'modissa'
    modis_container_name = 'modis-006'
    modis_account_url = 'https://' + modis_account_name + '.blob.core.windows.net/'
    modis_blob_root = modis_account_url + modis_container_name + '/'

    os.makedirs(args.modis_dir, exist_ok=True)

    modis_container_client = ContainerClient(account_url=modis_account_url,
                                             container_name=modis_container_name,
                                             credential=None)

    hvs = {(8,4),(8,5),(9,4),(9,5),(10,4),(10,5)}

    # Start loading
    for day in tq.tqdm(date_range):
        for (h, v) in hvs: # iterate throw H-V chunks
            folder = "MYD10A1" + '/' + '{:0>2d}/{:0>2d}'.format(h,v) + '/' + f"{day:%Y%j}"
            # Find all HDF files from this tile on this day
            filenames = list_hdf_blobs_in_folder(
                    modis_container_name, folder, modis_container_client)
            # Work with the first returned URL
            if len(filenames) > 0:
                blob_name = filenames[0]
                url = modis_blob_root + blob_name
                filename = os.path.join(args.modis_dir, blob_name)
                if not os.path.isfile(filename):
                    Path(filename).parent.mkdir(parents=True, exist_ok=True)
                    wget.download(url, filename)

def loader(url, filename, timeout=1.):
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
    except requests.exceptions.ConnectTimeout:
        return False
    if resp.ok:
        with open(filename, 'wb') as f:
            f.write(resp.content)
        return True
    else:
        return False

def download_hrrr(date_range, args):

    # Constants for creating the full URL
    blob_container = "https://noaahrrr.blob.core.windows.net/hrrr"
    container_aws = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
    container_google = "https://storage.googleapis.com/high-resolution-rapid-refresh"

    sector = "conus"
    forecast_hour = 0   # offset from cycle time
    product = "wrfsfcf" # 2D surface levels

    for day in tq.tqdm(date_range):

        # t00 - cycle
        for cycle in range(0, 3):
            # Put it all together
            file_path = f"hrrr.t{cycle:02}z.{product}{forecast_hour:02}.grib2"
            parent = os.path.join(args.hrrr_dir, f"hrrr.{day:%Y%m%d}/{sector}")
            filename = os.path.join(parent, f"{file_path}")

            if os.path.isfile(filename): break
            os.makedirs(parent, exist_ok=True)
            url = f"{blob_container}/hrrr.{day:%Y%m%d}/{sector}/{file_path}"
            flag = loader(url, filename, timeout=1.)
            if flag: break

        ##### CHECK #####
        if not os.path.isfile(filename):
            print(f"Not found: {filename}")

        # t12 - cycle
        for cycle in range(12, 9, -1):
            # Put it all together
            file_path = f"hrrr.t{cycle:02}z.{product}{forecast_hour:02}.grib2"
            parent = os.path.join(args.hrrr_dir, f"hrrr.{day:%Y%m%d}/{sector}")
            filename = os.path.join(parent, f"{file_path}")

            if os.path.isfile(filename): break
            os.makedirs(parent, exist_ok=True)
            url = f"{blob_container}/hrrr.{day:%Y%m%d}/{sector}/{file_path}"
            flag = loader(url, filename, timeout=1.)
            if flag: break

        ##### CHECK #####

        if not os.path.isfile(filename):
            print(f"Not found: {filename}")

def load_static(args):
    grid_cells = gpd.read_file(args.grid_cells)

    demtiff = rio.open(args.dem_file)
    soiltif = rio.open(args.soil_file)
    images_dem = []
    images_soil = []
    bins = np.array([0, 1, 2, 3, 4, 10, 14, 20, 29, 39, 49, 59, 69, 79, 84, 94])

    for idx, row in grid_cells.iterrows():

        image_dem = demtiff.read(1,
                window=demtiff.window(*row.geometry.bounds), out_shape=(10,10))
        images_dem.append(image_dem)

        image_soil = soiltif.read(1,
                window=soiltif.window(*row.geometry.bounds), out_shape=(10,10))
        image_soil = np.digitize(image_soil, bins, right=True)
        images_soil.append(image_soil)

    images_dem = np.stack(images_dem).astype(np.float32)
    images_soil = np.stack(images_soil).astype(np.int64)
    return images_dem, images_soil

def load_hrrr(date_range, args):

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

    tmp_dir = tempfile.TemporaryDirectory()
    num_chunks = np.ceil(len(date_range) / 31).astype(int)

    for i, dates in enumerate(tq.tqdm(np.array_split(date_range, num_chunks))):
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

def load_modis(date_range, args):

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
             data_dir = args.modis_dir) for day in tq.tqdm(date_range)],
        dim='time').transpose("time", "cell_id", "x", "y")

    return modis_ds.compute()

def round_time(ds):
    ds.coords['time'] = ds.coords['time'].dt.floor('D')
    return ds

def get_points(date_range,
               proj_x, proj_y, mid_x, mid_y, hrrr_dir):
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

    ds = xr.merge([
        # Temperature T12
        xr.open_mfdataset(fnamest12, engine='cfgrib',
                             backend_kwargs={'indexpath':''},
                             drop_variables = ['latitude', 'longitude', 'valid_time', 'step'],
                             filter_by_keys={'stepType': 'instant',
                             'typeOfLevel': 'surface', 'shortName': 't'},
                             preprocess = round_time,
            concat_dim='time', combine='nested', parallel=True).rename({'t': 't12'}),
        # U component of wind
        xr.open_mfdataset(fnamest12, engine='cfgrib',
                             backend_kwargs={'indexpath':''},
                             drop_variables = ['latitude', 'longitude', 'valid_time', 'step'],
                             filter_by_keys={
                             'stepType': 'instant',
                             'typeOfLevel': 'heightAboveGround',
                             'shortName': 'u'},
                             preprocess = round_time,
                             concat_dim='time', combine='nested', parallel=True),
        # V component of wind
        xr.open_mfdataset(fnamest12, engine='cfgrib',
                             backend_kwargs={'indexpath':''},
                             drop_variables = ['latitude', 'longitude', 'valid_time', 'step'],
                             filter_by_keys={
                             'stepType': 'instant',
                             'typeOfLevel': 'heightAboveGround',
                             'shortName': 'v'},
                             preprocess = round_time,
                             concat_dim='time', combine='nested', parallel=True),
        # Water equivalent of accumulated snow depth
        xr.open_mfdataset(fnamest12, engine='cfgrib',
                             backend_kwargs={'indexpath':''},
                             drop_variables = ['latitude', 'longitude', 'valid_time', 'step'],
                             filter_by_keys={'stepType': 'instant',
                             'typeOfLevel': 'surface', 'shortName': 'sdwe'},
                             preprocess = round_time,
                             concat_dim='time', combine='nested', parallel=True),
        # Precipitable water
        xr.open_mfdataset(fnamest12, engine='cfgrib',
                             backend_kwargs={'indexpath':''},
                             drop_variables = ['latitude', 'longitude', 'valid_time', 'step'],
                             filter_by_keys={'stepType': 'instant',
                                'typeOfLevel': 'atmosphereSingleLayer',
                                'shortName': 'pwat'},
                             preprocess = round_time,
                             concat_dim='time', combine='nested', parallel=True),
        # Maximum/Composite radar reflectivity
        xr.open_mfdataset(fnamest12, engine='cfgrib',
                             backend_kwargs={'indexpath':''},
                             drop_variables = ['latitude', 'longitude', 'valid_time', 'step'],
                             filter_by_keys={'stepType': 'instant',
                                'typeOfLevel': 'atmosphere',
                                'shortName': 'refc'},
                             preprocess = round_time,
                             concat_dim='time', combine='nested', parallel=True),
        # Temperature T00
        xr.open_mfdataset(fnamest00, engine='cfgrib',
                             backend_kwargs={'indexpath':''},
                             drop_variables = ['latitude', 'longitude', 'valid_time', 'step'],
                             filter_by_keys={'stepType': 'instant',
                             'typeOfLevel': 'surface', 'shortName': 't'},
                             preprocess = round_time,
            concat_dim='time', combine='nested', parallel=True).rename({'t': 't00'}),

        # Water equivalent of accumulated snow depth - Day accumulated
        xr.open_mfdataset(fnamest00, engine='cfgrib',
                             backend_kwargs={'indexpath':''},
                             drop_variables = ['latitude', 'longitude', 'valid_time', 'step'],
                             filter_by_keys={
                             'stepType': 'accum',
                             'typeOfLevel': 'surface',
                             'shortName': 'sdwe'},
                             preprocess = round_time,
            concat_dim='time', combine='nested', parallel=True).rename({'sdwe': 'sdwea'})
    ]).reindex({'time': date_range})

    ds['x'] = proj_x
    ds['y'] = proj_y

    points = ds.sel(x=mid_x, y=mid_y, method="nearest")
    del ds
    return points.compute()

def flatten(outter):
    return [item for sublist in outter for item in sublist]

def get_data(day, xs, ys, rx, ry,
             cell_id, product, variable, data_dir):

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
