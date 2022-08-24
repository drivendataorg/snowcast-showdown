## Environment
Creaated with conda from file environment.yml:
>conda create env -f environment.yml

Don't forget activate environment
>conda activate snow

All code must be run from `code` directory.
## Directory structure:
```
.code
├── Dataset Pipeline.ipynb
├── Evaluation-UpdateProcess.ipynb
├── Evaluation.ipynb
├── Train Model.ipynb
├── dataset.py
├── evaluation.py
├── hrrr_filenames.txt
├── hrrr_sample.grib2
├── layers.py
├── sam.py
├── submission.csv
├── sun_decline.csv
├── sun_decline.py
├── data
│   ├── hrrr
│   │   ├── ...
│   │   └── hrrr.20220210
│   │       └── conus
│   │           ├── hrrr.t00z.wrfsfcf00.grib2
│   │           └── hrrr.t12z.wrfsfcf00.grib2
│   ├── copernicus_dem
│   │   ├── COP90.tif
│   │   └── ...
│   ├── global_soil_regions
│   │   └── so2015v2.tif
│   ├── modis
│   │   └── MYD10A1
│   │       ├── 08
│   │       │   ├── 04
│   │       │   │   └── 2022038
│   │       │   │       └── MYD10A1.A2022038.h08v04.006.2022040043929.hdf
│   │       │   └── 05
│   │       │       └── 2022038
│   │       │           └── MYD10A1.A2022038.h08v05.006.2022040043303.hdf
│   │       ├── 09
│   │       │   └── ...
│   │       └── 10
│   │           └── ...
│   ├── dem.py
│   ├── hrrr_t00.py
│   ├── hrrr_t12.py
│   └── modis.py
├── development
│   ├── grid_cells.geojson
│   ├── hrrr
│   │   └── ...
│   ├── modis
│   │   └── ...
│   ├── labels_2020_2021.csv
│   ├── submission_format.csv
│   ├── train_dataset.nc
│   └── train_labels.csv
├── evaluation
│   ├── evaluation_dataset.nc
│   ├── grid_cells.geojson
│   ├── submission.csv
│   └── submission_format.csv
├── runs
│   ├── SnowNet_Fold#00
│   │   └── ...
│   └── ...
└── weights
    ├── SnowNet_fold_0_best.pt
    ├── ...
    └── SnowNet_fold_4_last.pt
```
## Data used for evaluation:
- [MYD10A1](https://nsidc.org/data/MYD10A1) - data access location: `https://modissa.blob.core.windows.net/modis-006`
- [HRRR](https://rapidrefresh.noaa.gov/hrrr/) - data access location: `https://noaahrrr.blob.core.windows.net/hrrr`
- [Copernicus DEM (90 meter resolution)](https://object.cloud.sdsc.edu/v1/AUTH_opentopography/www/metadata/Copernicus_metadata.pdf) - data access location: `https://planetarycomputer.microsoft.com/api/stac/v1/collections/cop-dem-glo-90`
- [FAO-UNESCO Global Soil Regions Map](https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/use/?cid=nrcs142p2_054013) - data access location: `s3://drivendata-public-assets/soil_regions_map.tar.gz`
- Information about sun daylight duration - Calculus based on date and latitude. Formulas from https://gml.noaa.gov/grad/solcalc/calcdetails.html.

Data stored as in the remote blob/bucket for convenience.

From HRRR used [features](https://www.nco.ncep.noaa.gov/pmb/products/hrrr/hrrr.t00z.wrfsfcf00.grib2.shtml):

t12z:
- 001 	entire atmosphere 	REFC 	analysis 	Composite reflectivity [dB]
- 060 	80 m above ground 	UGRD 	analysis 	U-Component of Wind [m/s]
- 061 	80 m above ground 	VGRD 	analysis 	V-Component of Wind [m/s]
- 064 	surface 	TMP 	analysis 	Temperature [K]
- 068 	surface 	WEASD 	analysis 	Water Equivalent of Accumulated Snow Depth [kg/m^2]
- 107 	entire atmosphere (considered as a single layer) 	PWAT 	analysis 	Precipitable Water [kg/m^2]

t00z:
- 064 	surface 	TMP 	analysis 	Temperature [K]
- 085 	surface 	WEASD 	0-0 day acc f 	Water Equivalent of Accumulated Snow Depth [kg/m^2]

`hrrr_sample.grib2` - just file with any single variable for faster loading grid data for HRRR files.

From MYD10A1 used the raw NDSI values.

### Download data
Data downloaded with scripts in folder `data`.
Example usage:
> python data/modis.py --data_dir data/modis --start_date 2014-09-01 --end_date 2021-07-01

> python data/hrrr_t00.py --data_dir data/hrrr --start_date 2014-09-01 --end_date 2021-07-01

> python data/hrrr_t12.py --data_dir data/hrrr --start_date 2014-09-01 --end_date 2021-07-01

File included to zip file
> python data/dem.py --data_dir data/copernicus_dem --grid_cells development/grid_cells.geojson  --output_file COP90.tif

Global Soil Regions Map also included.

## Train process
### Create dataset
Dataset created with `Dataset Pipeline.ipynb`. Make sure you download all files before start this notebook.
The process is highly dependent on the speed of reading data.
Output file included to zip file (train_dataset.nc).

### Train model
Training process - `Train Model.ipynb`. Information about loss, hparam etc. gathering with tensorboard and saved in folder `runs`.
`weights` - folder where stored weights for models.

## Evaluation
Evaluation represents with files `Evaluation.ipynb` (equivalent `evaluation.py`) and `Evaluation-UpdateProcess.ipynb`.

Both scripts can download files from blob (HRRR and MODIS) if its required.

`Evaluation.ipynb` - It creates a csv file with one column for the specified date. Can download files.

`Evaluation-UpdateProcess.ipynb` - It update a csv file submission_format.csv for the specified column/date in file header. Can download files and save intermediate dataset for future prediction. If dataset already exist - it will be updated.

Files content this parameters, where date this target date for evaluation and grid_cells contain geo information about locations.
```
args = argparse.Namespace(
    date = '2022-01-13',
    timelag = 92,
    hrrr_dir ='data/hrrr',
    modis_dir = 'data/modis',
    grid_cells ='development/ground_measures_features.geojson',
    dem_file = 'data/copernicus_dem/COP90_hh.tif',
    soil_file = 'data/global_soil_regions/so2015v2.tif',
    hrrr_sample = 'hrrr_sample.grib2',
    sun_decline = 'sun_decline.csv',
    model_dir = 'weights',
    dataset_file = 'development/evaluation_snotel.nc',
    format_file ='development/subm.csv',
    output_file ='development/subm.csv',
)
```
### Pipeline
0. Download files. (See "Download data")
1. Build dataset - `Dataset Pipeline.ipynb`
2. Train model - `Train Model.ipynb`
3. Evaluate `Evaluation-UpdateProcess.ipynb` or `Evaluation.ipynb`.
  - Make sure you have a tif file with sufficient coverage

### System
memory:         16GiB System memory

processor:      AMD Ryzen 5 5600X 6-Core Processor

GPU not used
