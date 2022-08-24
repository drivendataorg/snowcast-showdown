# 2nd place Solution - UltimateHydrology Team

Evgeniy Malygin, Maxim Kharlamov, Ivan Malygin, Maria Sakirkina, Ekaterina Rets

## Summary

Here we present combined physically-based and machine learning approaches to SWE (snow water equivalent)
prediction based on a wide range of data 
(in-situ, remote sensing data, results of general circulation modeling) 
using different SOTA implementations of Gradient Boosting Machine algorithm: 
XGBoost, LightGBM, and CatBoost, and their ensembles. Here are the main results of our research.
First of all, as a result of feature research and feature engineering, a dataset of 120 features was created, 
describing the physics of the process, as well as natural-climatic features of the region under study. Next,
we trained and tested an ML model for SWE forecast in the Colorado and Sierra Nevada Mountains (USA), 
using different implementations of the Gradient Boosting technique. Besides, the robustness of the model was 
checked by validation and hidden sampling and confirmed by low RMSE metric values (3.66) on the public 
leaderboard and 3.97 on the private leaderboard (2nd place). Moreover, we developed an end-to-end solution and automated real-time forecast pipeline to 
reproduce the forecast for each week. Hence, the developed solution is universal and can be used both 
for operational monitoring and for forecasting in advance.


# Setup

In the development we used Conda package manager, the other managers were not tested:
* Miniconda https://docs.conda.io/en/latest/miniconda.html
* conda version : 4.11.0
* Python 3.9.5.final.0

For Windows 10 and Linux, the attached file environments.yml contains all necessary libraries and dependencies,
to install them please use:
```commandline
conda update conda=4.11.0
conda env create --name ultimatehydrology python=3.9 --file=environment.yml
conda activate ultimatehydrology
```

Mac OS was not tested.

# Hardware
We used 3 workstations for data analysis, feature engineering, model training and experiments.
All calculations were performed on the CPU, the GPU was not used. 
Specifications are below:

* Workstation 1: AMD Ryzen Threadripper 1950X 16/32, 64 Gb RAM
* Workstation 2: AMD Ryzen AMD Ryzen 9 5950X 16/32, 32 Gb RAM
* Workstation 3: Intel i7 11800H 8/16, 16 Gb RAM

Train and inference time:
* Collecting DEM features: about 1 hour on Workstation 2
* Collecting train dataset time: about 27 hours on Workstation 3
* Training time: about 30 min on Workstation 2
* Inference time: about 10 min on Workstation 2 for each submission date

# Data sources used in training and inference

Most of the features (attributes) can be classified into 4 main groups according to the different data sources.
All of them are used in the training and inference code.

## 1. Meteorological parameters 
Meteorological parameters and their aggregates for 3 time periods from atmospheric model High-Resolution Rapid Refresh (HRRR NOAA):
* Average for the day of the forecast
* Aggregates by weekly period (week before the date of the forecast)
* Aggregates by winter period (from December of the previous year)

The High-Resolution Rapid Refresh (HRRR) is a NOAA real-time 3km resolution, hourly updated, cloud-resolving, convection-allowing atmospheric model, initialized by 3km grids with 3km radar assimilation.
Data is updated weekly for historical and future predictions. 

We connected to the Approved data access location: https://noaahrrr.blob.core.windows.net/hrrr 

## 2. MODIS Terra MOD10A1 satellite imagery product (Snow Cover Daily L3 Global 500m SIN Grid)

The snow cover algorithm calculates NDSI for all land and inland water pixels in daylight using MODIS band 4 (visible green) and band 6 (shortwave near-infrared). 
 
We used three main features:
* NDSI_Snow_Cover
* NDSI
* Snow_Albedo_Daily_Tile

We connected to Approved data access location: https://modissa.blob.core.windows.net/modis-006 

## 3. DEM Features
Relief parameters, produced on Copernicus DEM.
Relief parameters were calculated by usage of Copernicus DEM for each grid 
cell, as well as their aggregates in 200 and 500 m geodetic 
buffers (min, max, average, median) with the opensource libraries (GDAL, RichDEM). 
These attributes were calculated in advance for the whole grid and do not change over time. 

To create DEM features run all cells in dem_features_processing.ipynb.

We used Approved data access location: 
https://planetarycomputer.microsoft.com/api/stac/v1/collections/cop-dem-glo-90 

Runtime: about 1 hour on Workstation 2

Default save path: data/raw/dem_features.csv

Download DEM features dataset: https://disk.yandex.ru/d/3Ixxz7VfSEyMNA

## 4. Additional sources

* Coordinates of cell centroids
* Height in the cell centroid 
* Ordinal date from the beginning of the year (Day of year)
* Spatial-temporal interpolation based on Ground measure data (SNOTEL, CDEC) by Random Forest model. 4 parameters are used:
  * lat
  * lon 
  * alt 
  * time (day_of_year)

# Create train dataset

To create train dataset run all cells in train_dataset.ipynb. 

Main steps:
1. Downloading and processing the meteorological data (HRRR) for train period (from Dec 2014 to Jun 2021)
2. Downloading and processing of MODIS satellite images for train period (from Dec 2014 to Jun 2021)
3. Create HRRR weekly features (see more in model report: reports/Model report UltimateHydrology.pdf)
4. Create HRRR winter features (see more in model report: reports/Model report UltimateHydrology.pdf)
5. Create MODIS features (see more in model report: reports/Model report UltimateHydrology.pdf)
6. Join DEM features
7. Save train dataset 

Default save path: data/raw/dataset_train.csv

Collecting train dataset time: about 27 hours on Workstation 3

Requirements: 270 Gb free disk space (for raw HRRR and MODIS data and preprocessed files)

Download train dataset: https://disk.yandex.ru/d/fVVcLFoLMRdKIw

# Run training
To run the training model pipeline you just need to run all 
cells in train_model.ipynb in the created environment. 

* Model weights was saved out to by default: models/models_final.pkl
* Model weights file require 2.7 Gb
* Download model weights: https://disk.yandex.ru/d/ssM2CKEpurtHIA  

# Run inference
We provide a main point of entry to our code as the Jupyter Notebook that runs all steps of the 
pipeline to run inference source code and model weights. 

Brief description of the model. The solution uses gradient boosting models and their stacking:
1.	First level models:
- XGBRegressor – CV Score 5 Folds RMSE = 3.75
- CatBoostRegressor – CV 5 Folds Score RMSE = 3.65
- LGBMRegressor – CV 5 Folds Score RMSE = 3.78
2.	Second level model (meta-model):
- XGBRegressor – CV 5 Folds Score RMSE = 3.49
## Running the model

To run the weekly forecast you just need to run main_runner.ipynb in the created environment. 
```commandline
ipython kernel install --user --name= ultimatehydrology
```

Running the steps of model:
1.	Downloading and processing the meteorological data for the previous week
2.	Downloading and processing of MODIS space images for the previous week
3.	Data collection from new (meteo, space images) and stable(relief, etc.) resources
4.	Loading model weights from the pretrained model
5.	Making prediction and preparing a weekly submission for appropriate week.
