# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 21:49:29 2022

@author: Kharl
"""

import pandas as pd
import numpy as np

from tqdm.auto import tqdm
import os

from datetime import datetime

from joblib import Parallel, delayed


def cell_id_f(sample, dates_str):
    dates = pd.to_datetime(dates_str)
    dates_m1 = dates - pd.Timedelta(1, "d")
    dates_m7 = dates - pd.Timedelta(7, "d")  # недельный лаг

    idd = sample['cell_id'].unique()[0]
    sample['valid_time'] = pd.to_datetime(sample['valid_time'])
    sample = sample.set_index('valid_time')
    sample = sample.resample('H').mean()
    sample = sample.interpolate()

    sample['t2m_pls'] = np.where(sample['t2m'] >= 0, sample['t2m'], 0)
    sample['t2m_mns'] = np.where(sample['t2m'] < 0, sample['t2m'], 0)

    sample['tp_pls'] = np.where(sample['t2m'] >= 0, sample['tp'], 0)
    sample['tp_mns'] = np.where(sample['t2m'] < 0, sample['tp'], 0)

    sample['rain_en'] = sample['tp_pls'] * sample['t2m_pls']

    s2 = sample[['si10', 'dswrf', 't2m']].resample('D').mean()
    s2[['tp', 'tp_pls', 'tp_mns', 't2m_pls', 't2m_mns', 'rain_nrg']] = sample[
        ['tp', 'tp_pls', 'tp_mns', 't2m_pls', 't2m_mns', 'rain_en']].resample('D').sum()

    for col in s2.columns:
        s2[col + '_cumsum'] = s2[col].cumsum()
        s2[col + '_mean_sws'] = s2[col + '_cumsum'] / range(1, len(s2) + 1)

    features = s2[s2.index.isin(dates_m1)].copy()

    features['valid_time'] = dates_str
    features['cell_id'] = idd
    features = features.set_index(['cell_id', 'valid_time'])

    features2 = s2[s2.index.isin(dates_m7)].copy()
    features2['valid_time'] = dates_str
    features2['cell_id'] = idd
    features2 = features2.set_index(['cell_id', 'valid_time'])
    features2 = features.add_suffix('_m7')

    features = pd.concat([features, features2], axis=1)

    return features


def applyParallel(dfGrouped, func, dates_str):
    retLst = Parallel(n_jobs=-1)(delayed(func)(group, dates_str) for name, group in tqdm(dfGrouped))
    return pd.concat(retLst)


def winter_features(path_timeseries, features_save_path, last_year=False):
    cut = 0
    if last_year: cut = -1

    f = []
    for p in tqdm(os.listdir(path_timeseries)[cut:]):  # year walk
        path = os.path.join(path_timeseries, p)

        files = sorted(os.listdir(path))  # файлы в для данного года

        # make dates from file names
        dates_str = [file[:-4] for file in files]

        # concat files for year
        s1 = datetime.now()
        df = pd.concat([pd.read_csv(os.path.join(path, file)) for file in files])
        print('concat: ', datetime.now() - s1)

        df['t2m'] = df['t2m'] - 273.15

        # drop_duplicates
        s1 = datetime.now()
        df = df.set_index(['valid_time', 'cell_id'])
        df = df[~df.index.duplicated(keep='first')]
        df = df.reset_index()
        print('drop_dubl: ', datetime.now() - s1)

        # features
        s1 = datetime.now()
        features = applyParallel(df.groupby('cell_id'), cell_id_f, dates_str)
        print(datetime.now() - s1)

        for d in dates_str:
            if not os.path.exists(features_save_path):
                os.makedirs(features_save_path)

            (features
             .iloc[features.index.get_level_values('valid_time') == d]
             .to_csv(os.path.join(features_save_path, d + '.csv'))
             )
