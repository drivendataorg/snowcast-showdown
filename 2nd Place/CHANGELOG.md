# Changelog

Reflects code changes over the course of the evaluation stage of the competition. The final code is contained in this repo.

## Changes 2020-03-03 -- 2022-03-24

### 0. Model weights didn't change
According to the Competition Rules, we did not change the model weights. 
All changes relate to the inference code and account for changes in the incoming data streams.

### 1. File main_runner.ipynb:
We faced with problem that MODIS satellite images did not arriving on the submission day. 
We had to take satellite images for previous dates, or make a tiles composition for different dates. 
We realized that manually fixing the code is a bad way, so we fixed it in future versions of the code and made automatic processing for MODIS satellite images. 
* Cell 13 add lines 10-25:
 ```py
df_modis = df_modis[df_modis['valid_time']!='2022-03-03']
df_modis = df_modis[df_modis['valid_time']!='2022-03-17']
df_modis = df_modis[df_modis['valid_time']!='2022-03-24']
df_modis['valid_time'] = df_modis['valid_time'].str.replace('2022-03-02', '2022-03-03')
df_modis['valid_time'] = df_modis['valid_time'].str.replace('2022-03-23', '2022-03-24')
grid_cells['lon'] = grid_cells.centroid.geometry.x
grid_cells['lat'] = grid_cells.centroid.geometry.y
df_modis = df_modis.merge(grid_cells[['cell_id', 'lon', 'lat']], on='cell_id', how='inner')
grid_cells = grid_cells.drop(['lat', 'lon'], axis=1)
df_modis_1703 = pd.concat([df_modis[(df_modis['valid_time'] == '2022-03-16') & (df_modis['lon'] < -117)], 
                           df_modis[(df_modis['valid_time'] == '2022-03-14') & (df_modis['lon'] > -117)]])
df_modis_1703['valid_time'] = '2022-03-17'
df_modis_1703 = df_modis_1703.drop(['lat', 'lon'], axis=1)
df_modis = df_modis.drop(['lat', 'lon'], axis=1)
df_modis = pd.concat([df_modis, df_modis_1703])
```

### 2. File src/data/modis_downloader.py:
* line 7: change valiable "temporalr" (manual dates list for submission dates) to download MODIS satellite images for each day:
```py
temporalr = [str(i) for i in range(2022001, 2022182)]
```

## Changes 2022-03-31 -- 2022-04-07

### 0. Model weights didn't change
According to the Competition Rules, we did not change the model weights. 
All changes relate to the inference code and account for changes in the incoming data streams.

### 1. File main_runner.ipynb:
We add automatic processing for MODIS satellite images and save previous changes (before current update).
* Cell 13 add lines 27-46:
 ```py
# Changes after 2022-03-03
# Automatic processing for MODIS satellite images:
# get all images for previous week and get the last valid pixel value.

def modis_select(gr):
    gr_cloudless = gr[gr['sc'] <= 100]
    if len(gr_cloudless) != 0:
        g = gr_cloudless.head(1)
    else:
        g = gr.head(1)
    return g

df_modis['valid_time'] = pd.to_datetime(df_modis['valid_time'])
modis_after = []
modis_before = df_modis[df_modis['valid_time'] <=  pd.to_datetime('2022-03-24')]
forecast_dates = [pd.to_datetime(d) for d in dates if pd.to_datetime(d) > pd.to_datetime('2022-03-24') ]


for forecast_date in tqdm(forecast_dates):


    modis_forecast = df_modis[(df_modis['valid_time'] > forecast_date - timedelta(days=7)) &
                              (df_modis['valid_time'] <=forecast_date)]
    modis_forecast = modis_forecast.sort_values('valid_time', ascending=False).groupby('cell_id').apply(modis_select)
    modis_forecast = modis_forecast.reset_index(drop=True)
    modis_forecast['valid_time'] = forecast_date
    modis_after.append(modis_forecast)
modis_after = pd.concat(modis_after)    
df_modis = pd.concat([modis_before, modis_after])
```

## Changes 2022-04-14 -- 2022-06-30

### 0. Model weights didn't change

According to the Competition Rules, we did not change the model weights. 
All changes relate to the inference code and account for changes in the incoming data streams.

### 1. File src/features/modis_features.py:

* Add functions to automate processing of MODIS satellite images.
Collect all images and all tiles for previous week (week before submission date) 
and get last valid pixel value:

 ```py
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
```

### 2. File main_runner.ipynb:
The main goal of this code version is to structure and rearrange cells in main_runner.ipnb  

#### 1.  Imports
* Cell 1: structured imports

#### 2.  Config
* Cell 2: structured submissuon dates
* Cell 3: paths (not changed)
* Cell 4: move cell with feature columns (change cells order)  to block 2. Config
* Cell 5: read grid (not changed)

#### 3. Data Processing
* Cell 6: download Weekly HRRR Data (code not changed)
* Cell 7: download MODIS sattelite images (code not changed, but we changed cell order)
* Cell 8: Collect weekly HRRR Features (code not changed)
* Cell 9: Collect winter HRRR Features (code not changed)
* Cell 10: Prepare MODIS data: extract pixel values for grid centroid from .hdf file and save to .csv (code not changed)
* Cell 11: Collect MODIS features with automatic processing (see p.1  and src/features/modis_features.py):
```py
paths_modis = list(Path(PATH_MODIS_FEATURES).rglob('*.csv'))
df_modis = modis_features.get_modis_df(paths_modis, dates)
df_modis['valid_time'] = df_modis['valid_time'].dt.date.astype(str)
```

#### 4 Make inference dataset 
* Cell 12: Load and prepare submission format file:
```py
sub = pd.read_csv('data/input/submission_format_2022.csv')
sub = sub.melt('Unnamed: 0').fillna(0)
sub.columns = ['cell_id', 'valid_time', 'swe']
sub = sub.set_index(['valid_time', 'cell_id'])
```
* Cell 13: Load features datasets:
```py
paths_week_f = list(Path(PATH_WEEK_FEATURES).rglob('*.csv'))
df_week = [pd.read_csv(p) for p in paths_week_f]
df_week = pd.concat(df_week)

paths_winter_f = list(Path(PATH_WINTER_FEATURES).rglob('*.csv'))
df_winter = [pd.read_csv(p) for p in paths_winter_f]
df_winter = pd.concat(df_winter)

df_dem = pd.read_csv('data/raw/dem_features.csv')

df_grid = grid_cells.copy()
df_grid = gpd.GeoDataFrame(df_grid, geometry=df_grid.centroid)
df_grid['lon'] = df_grid.geometry.x
df_grid['lat'] = df_grid.geometry.y
df_grid = df_grid[['cell_id', 'lon', 'lat']]
```
* Cell 14: join all datasets and prepare dataset for inference:
```py
df = sub.reset_index().copy()

df = df.merge(df_week, on=['cell_id', 'valid_time'], how='left')
df = df.merge(df_winter, on=['cell_id', 'valid_time'], how='left')
df = df.merge(df_dem, on=['cell_id'], how='left')
df = df.merge(df_grid, on=['cell_id'], how='left')
df = df.merge(df_modis, on=['cell_id', 'valid_time'])

df['dt_date'] = pd.to_datetime(df['valid_time'], format='%Y-%m-%d') 
df['dayofyear'] = df['dt_date'].dt.dayofyear
df['year'] = df['dt_date'].dt.year
df = df.drop(['dt_date'], axis=1)
df = df[cols]
```
#### 5 Run Model

* Cell 15: Read model weights (code not changed)
* Cell 16: Get model (code not changed)
* Cell 17: Predict, make submission and join results to submission format file:
```py
name=0
Z_meta_f=pd.DataFrame(columns=zoo_names, index=Z.index).fillna(value=0)

for model in zoo: 
    Z_meta_f[zoo_names[name]]=model.predict(Z)
    name+=1

for i in Z_meta_f.columns:
    Z_meta_f[Z_meta_f[i]<0]=0

res=pd.DataFrame(clf_meta.predict(Z_meta_f))
res.columns=['swe_pred']

res = pd.concat([df[['cell_id', 'valid_time']], res], axis=1).set_index(['valid_time', 'cell_id'])
sub.loc[sub.index.isin(res.index), 'swe'] = res['swe_pred']
res_pivot = sub.reset_index().pivot(index='cell_id', columns='valid_time', values='swe')

```

* Cell 18: save predict to csv-file:
```py
res_pivot.to_csv(f'sub_{str(datetime.now().date())}.csv', index=True)
```

### 3. train_dataset.ipynb

We add train_dataset.ipynb to show how we've collected train dataset. 
This notebook runs about 27 hours (download data (about 190 Gb), prepare features, and collect train dataset).
Requirements: 270 Gb free disk space
