# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 23:59:21 2022

@author: Kharl
"""

import io
import os
from datetime import date, timedelta, datetime

import xarray as xr
import requests
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
import matplotlib.pyplot as plt
# import cmocean

# Not used directly, but used via xarray
import cfgrib

import pandas as pd

from tqdm.auto import tqdm

from joblib import Parallel, delayed

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

import os



def get_urls_for_requests_am_week(forecast_date, sector = "conus"):
    '''
    Create list of URLs for requests to get NOAA HRRR data

    Parameters
    ----------
    forecast_date : str
        forecast date. 
    

    Returns
    -------
    urls: list
    List of URLs for requests

    '''
    
    # Constants for creating the full URL
    
    blob_container = "https://noaahrrr.blob.core.windows.net/hrrr"
    'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20150101/conus/hrrr.t00z.wrfsfcf01.grib2'
    forecast_hour = 1   # offset from cycle time
    product = "wrfsfcf" # 2D surface levels
#     product = "sfc" # 2D surface levels
    
    urls = [] # output list 
    dates = pd.date_range(pd.to_datetime(forecast_date) - pd.Timedelta(7, "d"), forecast_date)                                 #dates
    #dates = dates[dates.month.isin(months)]
    
    for date in dates[:-1]:
        for cycle in range(24):

#             file_path = f"hrrr.t{cycle:02}z.{product}{forecast_hour:02}.grib2"
            url = f'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date:%Y%m%d}/conus/hrrr.t{cycle:02}z.{product}{forecast_hour:02}.grib2'
            urls.append(url)
    
    return urls

def download_hhhr_data(url, params, output_path):
    
    
    
    session = requests.Session()

    retries = Retry(total=10,
                    backoff_factor=0.1,
                    status_forcelist=[ 500, 502, 503, 504 ])

    session.mount('http://', HTTPAdapter(max_retries=retries))

# s.get('http://httpstat.us/500')
    try:
        idx = session.get(f"{url}.idx").text
    except ConnectionError:
        return url
    if 'Error' in idx:
        return url
    else:
        idx = idx.splitlines()
        date = idx[0].split(':')[2].split('=')[-1]
        out = []
        # Grab needed parametrs.
        for p in params:
            # Pluck the byte offset from this line, plus the beginning offset of the next line
            param_idx = [l for l in idx if p in l][0].split(":")
            line_num = int(param_idx[0])
            range_start = param_idx[1] #
            # The line number values are 1-indexed, so we don't need to increment it to get the next list index,
            # but check we're not already reading the last line
            next_line = idx[line_num].split(':') if line_num < len(idx) else None
            range_end = next_line[1] if next_line else None

            headers = {"Range": f"bytes={range_start}-{range_end}"}
            try:
                resp = session.get(url, headers=headers, stream=True)
            except ConnectionError:
                return url

            if not os.path.exists(output_path): os.makedirs(output_path)
            
            filename = '{}_{}'.format(p.split(':')[0], date)
            filename = os.path.join(output_path, filename)
            
            
            with open(filename, 'wb') as f:
                f.write(resp.content)
            
def additional():
    train_labels = pd.read_csv('C:/projects/hackathons/snowcast/train_labels.csv')
    submission = pd.read_csv('C:/projects/hackathons/snowcast/submission_format.csv')
    train_features = pd.read_csv('C:/projects/hackathons/snowcast/ground_measures_train_features.csv')
    test_features = pd.read_csv('C:/projects/hackathons/snowcast/ground_measures_test_features.csv')
    metadata = pd.read_csv('C:/projects/hackathons/snowcast/ground_measures_metadata.csv')
    #grid_cells = gpd.read_file('snowcast/grid_cells.geojson')
    
    
    path = 'hrrr_data//'
    if not os.path.exists(path): os.makedirs(path)
    ts_exists = os.listdir(path)
    
    aaa = list(train_labels)[63:] + list(submission)[1:] + list(test_features)[1:] + list(train_features)[57:]
    
    aaa_set = set(aaa)
    
    ll = []
    for f in aaa_set:
        if f not in ts_exists:
            ll.append(f)
            
    return sorted(ll)


#dates = ['2021-12-11','2022-02-12']

def download_data(dates, output_path, historical=False):

    # OUTPUT_FOLDER = 'hrrr_data/'
    
    PARAMS = ['TMP:2 m above ground',
              'APCP:surface',
              'DSWRF:surface',
              'WIND:10 m above ground',
              'WEASD:surface'
              
              #'RH:2 m above ground', 
             ]
    
    if historical: dates = additional()
    
    for d in tqdm(dates[:]):    
        print(d)
        url = get_urls_for_requests_am_week(d, sector = "conus")
            
        for param in PARAMS:
            
            try:
                pass
                print('download data ', param, d)
                out = Parallel(n_jobs=-1)(delayed(download_hhhr_data)(i, [param], output_path + d) for i in tqdm(url))
            except:
                print(d, param, 'download fail')













