# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 23:05:45 2022

@author: Kharl
"""

import os
import xarray as xr
import pandas as pd
import geopandas as gpd
from tqdm.auto import tqdm
from datetime import timedelta


def nearest_points3(grid_cells, df, proj):
    grid_cells = grid_cells.set_geometry(grid_cells.centroid)  #

    grid_cells = grid_cells.to_crs(proj)

    df['geometry'] = gpd.points_from_xy(df['x'], df['y'])
    df1 = gpd.GeoDataFrame(df, geometry='geometry')

    # df1 = df1.set_crs('epsg:4326')      #

    grid_cells_mrg = grid_cells.sjoin_nearest(df1, how='inner')

    table = grid_cells_mrg[['cell_id', 'y', 'x']].set_index(['y', 'x'])
    return table


def data_extraction(df, table):
    df = df.set_index(['y', 'x'])
    df = df.loc[table.index.drop_duplicates()]
    return table.join(df, how='inner')


def concat_modis_day(files_day):
    aaa = []

    for f in files_day:
        rds = xr.open_mfdataset(f, engine='rasterio')

        proj = rds.rio.crs

        df = rds[['NDSI_Snow_Cover', 'NDSI', 'Snow_Albedo_Daily_Tile']].to_dataframe()
        df = df.reset_index()
        df['valid_time'] = rds.RANGEBEGINNINGDATE

        aaa.append(df[['y', 'x', 'NDSI_Snow_Cover', 'NDSI', 'Snow_Albedo_Daily_Tile', 'valid_time']])

    return pd.concat(aaa), proj


def modis_features(path, grid_cells, all_files=True, last_files=0, output_path='modis_features/'):
    files = os.listdir(path)
    files = [x for x in files if x.endswith('.hdf')]

    gr = [(int((file.split(".")[1][5:]))) for file in files]

    ddf = pd.DataFrame()
    ddf['gr'] = gr
    ddf['path'] = path
    ddf['file_name'] = files
    ddf['path_full'] = ddf['path'] + os.sep + ddf['file_name']

    # nearest points
    for day in tqdm(ddf['gr'].unique()[:1]):
        files_day = list(ddf[ddf['gr'] == day]['path_full'])

        df_day, proj = concat_modis_day(files_day)

        table = nearest_points3(grid_cells.copy(), df_day.copy(), proj)

    if all_files:
        last_files = 0

    for day in tqdm(ddf['gr'].unique()[-last_files:]):
        try:
            files_day = list(ddf[ddf['gr'] == day]['path_full'])

            df_day, proj = concat_modis_day(files_day)

            data_day = data_extraction(df_day, table)

            data_day = data_day.set_index(['cell_id', 'valid_time'])[['NDSI_Snow_Cover', 'NDSI', 'Snow_Albedo_Daily_Tile']]
            data_day = data_day.rename(columns={"NDSI_Snow_Cover": "sc", "NDSI": "ndsi1", 'Snow_Albedo_Daily_Tile': 'sa1'})

            data_day['ndsi1'] = data_day['ndsi1'] * 10000

            data_day['sc'] = data_day['sc'].fillna(200)
            data_day['ndsi1'] = data_day['ndsi1'].fillna(-32768)
            data_day['sa1'] = data_day['sa1'].fillna(250)

            save_path = os.path.join(output_path, path.split('/')[-1])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            data_day.to_csv(os.path.join(save_path, df_day['valid_time'].unique()[0] + '.csv'))
        except Exception:
            pass


def modis_select(gr):
    gr_cloudless = gr[gr['sc'] <= 100]
    if len(gr_cloudless) != 0:
        g = gr_cloudless.head(1)
    else:
        g = gr.head(1)
    return g


def get_modis_df(paths_modis, dates):
    df = [pd.read_csv(p) for p in paths_modis]
    df = pd.concat(df)
    df['valid_time'] = pd.to_datetime(df['valid_time'])
    modis_dataset = []

    for d in tqdm(dates):
        d = pd.to_datetime(d)
        tmp = df[(df['valid_time'] > d - timedelta(days=7)) & (df['valid_time'] <= d)]
        tmp = tmp.sort_values('valid_time', ascending=False).groupby('cell_id').apply(modis_select)
        tmp = tmp[tmp['sc'] > 0].reset_index(drop=True)
        tmp['valid_time'] = d
        modis_dataset.append(tmp)

    return pd.concat(modis_dataset)
