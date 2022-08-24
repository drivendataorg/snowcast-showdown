import os
import argparse
import wget
import requests

import pandas as pd
import numpy as np
from pathlib import Path

from datetime import datetime, date, timedelta
from tqdm.auto import tqdm

from loguru import logger

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

def main(args):

    logger.info(
    f"Start downloading in data ranges: from {args.start_date:} to {args.end_date:}")

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

    delta = end_date - start_date

    # Constants for creating the full URL
    blob_container = "https://noaahrrr.blob.core.windows.net/hrrr"
    blob_container_aws = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
    blob_container_google = "https://storage.googleapis.com/high-resolution-rapid-refresh"
    sector = "conus"
    forecast_hour = 0   # offset from cycle time
    product = "wrfsfcf" # 2D surface levels

    logger.info(f"Downloading begun")

    for i in tqdm(range(delta.days + 1)):
        day = start_date + timedelta(days=i)

        for cycle in range(0, 3):
            # Put it all together
            file_path = f"hrrr.t{cycle:02}z.{product}{forecast_hour:02}.grib2"
            parent = os.path.join(args.data_dir, f"hrrr.{day:%Y%m%d}/{sector}")
            filename = os.path.join(parent, f"{file_path}")

            if os.path.isfile(filename): break
            os.makedirs(parent, exist_ok=True)
            url = f"{blob_container_aws}/hrrr.{day:%Y%m%d}/{sector}/{file_path}"
            flag = loader(url, filename, timeout=1.)
            if flag: break

        ##### CHECK #####

        if not os.path.isfile(filename):
            logger.error(f"Not found: {filename}")

    logger.info("Downloading complited")
    ## End loader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/hrrr', help='Data dir')
    parser.add_argument('--start_date',
                default='2014-07-31', help='Start search from data')
    parser.add_argument('--end_date', default=None, help='End data for files')
    args = parser.parse_args()

    if args.end_date is None:
        args.end_date = f"{datetime.today() - timedelta(days=1):%Y-%m-%d}"

    main(args)
