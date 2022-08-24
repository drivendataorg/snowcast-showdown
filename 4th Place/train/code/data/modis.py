import os
import argparse
import wget

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

from datetime import datetime
from tqdm.auto import tqdm

from azure.storage.blob import ContainerClient

from loguru import logger


def lat_lon_to_modis_tile(lat, lon, modis_tile_extents):
    """
    Get the modis tile indices (h,v) for a given lat/lon

    https://www.earthdatascience.org/tutorials/convert-modis-tile-to-lat-lon/
    """

    found_matching_tile = False
    i = 0
    while(not found_matching_tile):
        found_matching_tile = lat >= modis_tile_extents[i, 4] \
        and lat <= modis_tile_extents[i, 5] \
        and lon >= modis_tile_extents[i, 2] and lon <= modis_tile_extents[i, 3]
        i += 1

    v = int(modis_tile_extents[i-1, 0])
    h = int(modis_tile_extents[i-1, 1])

    return h,v


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

def loader(args):

    logger.info(
    f"Start downloading in data ranges: from {args.start_date:} to {args.end_date:}")

    modis_account_name = 'modissa'
    modis_container_name = 'modis-006'
    modis_account_url = 'https://' + modis_account_name + '.blob.core.windows.net/'
    modis_blob_root = modis_account_url + modis_container_name + '/'
    modis_tile_extents_url = modis_blob_root + 'sn_bound_10deg.txt'

    products = ["MYD10A1"]

    # This file is provided by NASA; it indicates the lat/lon extents of each
    # MODIS tile.
    #
    # The file originally comes from:
    #
    # https://modis-land.gsfc.nasa.gov/pdf/sn_bound_10deg.txt

    # modis_tile_extents_url = modis_blob_root + 'sn_bound_10deg.txt'

    os.makedirs(args.data_dir, exist_ok=True)
    # fn = os.path.join(args.data_dir, modis_tile_extents_url.split('/')[-1])
    # if not os.path.isfile(fn):
    #     Path(fn).parent.mkdir(parents=True, exist_ok=True)
    #     wget.download(modis_tile_extents_url, fn)

    # Load this file into a table, where each row is (v,h,lonmin,lonmax,latmin,latmax)
    # modis_tile_extents = np.genfromtxt(fn, skip_header = 7,  skip_footer = 3)
    modis_container_client = ContainerClient(account_url=modis_account_url,
                                             container_name=modis_container_name,
                                             credential=None)

    # logger.info(f"Load grid cell info")
    # df = gpd.read_file(args.grid_cells)
    # Grab all required H-V chunks
    # hvs = set()

    # for name, row in tqdm(df.iterrows(), total=df.shape[0]):
    #
    #     lon = row.geometry.centroid.x
    #     lat = row.geometry.centroid.y
    #     h, v = lat_lon_to_modis_tile(lat, lon, modis_tile_extents)

        # hvs.add((h, v))

    hvs = {(8,4),(8,5),(9,4),(9,5),(10,4),(10,5)}

    logger.info(f"All hv conbinations: {hvs}")

    # Start loading
    days = pd.date_range(start=args.start_date, end=args.end_date).tolist()

    logger.info(f"Downloading begun")

    for day in tqdm(days):
        daynum = f"{day.year}{day.timetuple().tm_yday:03d}"

        for (h, v) in hvs: # iterate throw H-V chunks
            for product in products:  # iterate throw products

                folder = product + '/' + '{:0>2d}/{:0>2d}'.format(h,v) + '/' + daynum
                # Find all HDF files from this tile on this day
                filenames = list_hdf_blobs_in_folder(
                        modis_container_name, folder, modis_container_client)

                # Work with the first returned URL
                if len(filenames) > 0:
                    blob_name = filenames[0]
                    url = modis_blob_root + blob_name
                    filename = os.path.join(args.data_dir, blob_name)
                    if not os.path.isfile(filename):
                        Path(filename).parent.mkdir(parents=True, exist_ok=True)
                        wget.download(url, filename)

    logger.info("Downloading complited")
    ## End loader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/modis', help='Data dir')
    parser.add_argument('--grid_cells',
                default='development/grid_cells.geojson', help='grid_cells json file')
    parser.add_argument('--start_date',
                default='2014-09-01', help='Start search from data')
    parser.add_argument('--end_date', default=None, help='End data for files')
    args = parser.parse_args()

    if args.end_date is None:
        args.end_date = f"{datetime.today():%Y-%m-%d}"

    loader(args)
