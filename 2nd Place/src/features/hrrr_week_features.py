# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 17:07:31 2022

@author: Kharl
"""
from datetime import datetime
import xarray as xr
import os

import pandas as pd
import geopandas as gpd

from tqdm.auto import tqdm

from joblib import Parallel, delayed

import glob



def nearest_points(grid_cells, ds):
    '''
    finding nearest grib's y and x for grid_cells 

    Parameters
    ----------
    grid_cells : DataFrame
        grid_cells
    ds : xarray dataset
        hrrr grib file

    Returns
    -------
    table : GeoDataFrame
        DataFrame with nearest y and x for grid_cells
        index = ('y', 'x')
        cols = ['cell_id']

    '''
    
    df = ds.to_dataframe()[['latitude', 'longitude']]
    df['longitude'] = df['longitude'] - 360         # переводим  в нормальную долготу

    df['geometry'] = gpd.points_from_xy(df['longitude'], df['latitude'])
    df1 = gpd.GeoDataFrame(df, geometry = 'geometry')
    
    df1 = df1.set_crs('epsg:4326')      #
    
    df1 = df1.reset_index()

    grid_cells_mrg = grid_cells.sjoin_nearest(df1, how = 'inner', max_distance = 0.25)

    table = grid_cells_mrg[['cell_id', 'y', 'x']].set_index(['y','x'])
    return table

def data_extraction(file_path, table):
    '''
    extraction data from grib for (y, x) from table

    Parameters
    ----------
    file_path : str
        
    table : DataFrame
        DataFrame with indexes 
        index = ('y', 'x')
        cols = ['cell_id']

    Returns
    -------
    DataFrame
        extracted data 

    '''

    df1 = xr.open_dataset(file_path, engine = 'cfgrib').to_dataframe()
    df1 = df1.loc[table.index.drop_duplicates()]
    return table.join(df1[['valid_time', df1.columns[-1]]], how = 'left')

def concat_data(data):
    '''
    

    Parameters
    ----------
    data : list of DataFrames 
        as a result of data_extraction.

    Returns
    -------
    df : DataFrame
        index = valid_date
        cols = ['cell_id', vars]
        vars = ['t2m', 'tp', ....etc]

    '''
    # data from data_extraction()
    var = set([d.columns[-1] for d in data])
    
    df = []
    for v in var:
        l = pd.concat([d for d in data if d.columns[-1] == v])
        df.append(l.set_index(['valid_time', 'cell_id']))
        
    df = pd.concat(df, axis = 1)
    df = df.reset_index().set_index(['valid_time'])
    
    return df


def make_features_gr(sample_slice):
    '''
    feature calculation for selected cell_id. using in groubpy
    
    
    Parameters
    ----------
    sample_slice : DataFrame
        index = valid_date
        
        cols = ['cell_id', vars]
        vars = ['t2m', 'tp', ...etc]

    Returns
    -------
    df_feature_d : DataFrame
        Features. index = ('valid_time', 'cell_id')

    '''
    
    
    #sample_slice = sample_id[sample_id['cell_id'] == sample_id['cell_id'].iloc[0]]
    cell_id = sample_slice['cell_id'].iloc[0]
    
    
    sample_slice = sample_slice.resample('H').mean()    
    df_feature_d = pd.DataFrame()
    
    try:
        sample_slice['t2m'] = sample_slice['t2m'].interpolate()
        sample_slice['t2m'] = sample_slice['t2m'] - 273.15
        sample_slice['tp'] = sample_slice['tp'].fillna(sample_slice['tp'].mean())       
        
        pls = sample_slice[sample_slice['t2m'] >= 0]
        mns = sample_slice[sample_slice['t2m'] < 0]
        
        
        df_feature_d['temp_mean'] = [sample_slice['t2m'].mean()]                                 
        df_feature_d['temp_sum'] = [sample_slice['t2m'].sum()]
        df_feature_d['temp_sum_cold'] = [mns['t2m'].sum()]          
        df_feature_d['temp_sum_warm'] = [pls['t2m'].sum()]         
        df_feature_d['temp_sum_cold_hours'] = [mns['t2m'].count()]  
        df_feature_d['temp_sum_warm_hours'] = [pls['t2m'].count()] 
        
        df_feature_d['tp_mean'] = [sample_slice['tp'].mean()]
        df_feature_d['tp_sum'] = [sample_slice['tp'].sum()] 
        df_feature_d['tp_sum_liquid'] = [pls['tp'].sum()] 
        df_feature_d['tp_sum_solid'] = [mns['tp'].sum()] 
        
        
        rain_enrg = pls['t2m'] * pls['tp']
        df_feature_d['rain_enrg'] = [rain_enrg.sum()]        
        
        ####### оттепели
        df_ind = pd.DataFrame()
        df_ind['t2m'] = mns.index
        
        aaa = pd.DataFrame()    # датафрейм со сдвинутыми датами
        aaa['t2m_sh'] = mns['t2m'].shift(1).dropna().index
        df_ind = pd.concat([df_ind, aaa], axis = 1) # соединяем
        
        df_ind['dif'] = df_ind['t2m_sh'] - df_ind['t2m']
        df_ind['dif'] = df_ind['dif'].dt.total_seconds()/3600 - 1
        df_ind = df_ind[df_ind['dif'] > 0]
    
        df_feature_d['thaw_count'] = [len(df_ind)]    #Количество переходов через 0 градусов за наделю до даты прогноза            
    except:
        pass
    
    for col in sample_slice.columns:
        if col not in ['t2m', 'tp']:
            sample_slice[col] = sample_slice[col].interpolate()
            
            df_feature_d[col + '_mean'] = [sample_slice[col].mean()]
            df_feature_d[col + '_sum'] = [sample_slice[col].sum()]
            
            if col in ['sdwe']:
                df_feature_d[col + '_range'] = [sample_slice[col].values[-1] - sample_slice[col].values[0]]
                df_feature_d[col + '_last'] = [sample_slice[col].values[-1]]
                df_feature_d[col + '_first'] = [sample_slice[col].values[0]]
            
    df_feature_d['cell_id'] = [cell_id]
    df_feature_d['valid_time'] = [sample_slice.index[-1]]
    
    df_feature_d = df_feature_d.set_index(['cell_id', 'valid_time'])
    #df_feature_d.index = [sample_slice.index[-1]]
    return df_feature_d

def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs= -1)(delayed(func)(group) for name, group in tqdm(dfGrouped, desc = 'features'))
    return pd.concat(retLst)

def features_for_timestamp(folder,
                           grid_cells,
                           features_save_path,
                           save_extracted_data = True,
                           make_week_features = True,
                           save_path = 'hrrr_extracted_data\\'):
    '''
    main
    
    1. finding nearest points in gribs for grid_cells
    2. extraction data for all files in folder
    3. concatination data
    4. calculation of features

    Parameters
    ----------
    folder : DataFrame
        folder with files for timestamp
    grid_cells : TYPE
        file with grid_cells (grid_cells.geojson).

    Returns
    -------
    features : TYPE
        DESCRIPTION.

    '''
    if not os.path.exists(save_path): os.makedirs(save_path)
    
    st = datetime.now()
    
    files = glob.glob(folder + '*/*')
    remove = [os.remove(file) for file in files if file.endswith('.idx')]            # удаляем лишние idx
    files = [file for file in files if not file.endswith('.idx')]

    st1 = datetime.now()
    table = nearest_points(grid_cells, xr.open_dataset(files[0], engine = 'cfgrib', backend_kwargs={'indexpath':''}))

    st2 = datetime.now()
    data = Parallel(n_jobs=-1)(delayed(data_extraction)(file, table) for file in tqdm(files, desc = 'data_extraction'))
    print('from grib to df: ', datetime.now() - st2)
    
    st3 = datetime.now()
    df = concat_data(data)
    print('data concat: ', datetime.now() - st3)
    
    if save_extracted_data: 
        st4 = datetime.now()
        file_name = folder.split(os.sep)[-1]
        y = int(file_name[:4])
        m = int(file_name[5:7])
        if m > 8: y += 1

        out_path = os.path.join(save_path, str(y))
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        df.drop(['sdwe'], axis = 1).to_csv(os.path.join(out_path, f'{file_name}.csv'))        # save data
        print('save_extracted_data: ', datetime.now() - st4)
        

    
    if make_week_features:
        # features joblib
        st4 = datetime.now()
        
        features = applyParallel(df.groupby('cell_id'), make_features_gr)
        print('making features: ', datetime.now() - st4)
        
        print()
        print('total: ', datetime.now() - st)
        
        features = features.reset_index()
        features['valid_time'] = folder.split(os.sep)[-1]
        features = features.set_index(['cell_id', 'valid_time'])
        

        if not os.path.exists(features_save_path): os.makedirs(features_save_path)
        features.to_csv(os.path.join(features_save_path,folder.split(os.sep)[-1] + '.csv'))
    
    
