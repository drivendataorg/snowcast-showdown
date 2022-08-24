# Solution: Snowcast Showdown: Evaluation Stage

1st place with 3.8784 RMSE score.

## Installation

System requirements:

- Python 3.7-3.9
- CUDAToolkit 11.1+
- wget

How to resolve python requirements:
```
conda install -c conda-forge pyhdf
pip install torch==1.9.1 --extra-index-url https://download.pytorch.org/whl/cu111
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
pip install azure==4.0.0
pip install azure-storage-blob==12.9.0 # See issue https://github.com/Azure/azure-storage-python/issues/389
pip install -r requirements.txt
conda install -c conda-forge cartopy==0.20.0 proj==8.1.1 geos==3.9.1 hdf5==1.12.1 --yes
```

Alternative: using `dockerfile`

## The structure of the directory

The correct structure is:
```
snowcast                   <- the main directory
-data
--dem                      <- directory will be contains the DEM cache
--development              <- directory contains the training data 
---labels_2020_2021.csv*
---ground_measures_metadata.csv*
---submission_format.csv*
---grid_cells.geojson*
---train_labels.csv*
---ground_measures_test_features.csv*
---ground_measures_train_features.csv*
---modisclm.csv 	   <- calcs ~15 hours based on local MODIS Terra MOD10A1 and Aqua MYD10A1 archive
---grid.csv     	   <- grid_id's features cache
---stmeta.csv              <- stations's features cache
--evaluation               <- directory contains the inference data
---ground_measures_metadata.csv**
---submission_format.csv**
---grid_cells.geojson**
---submissionslist.csv     <- file will be contains the list of submissions
---modisclm.csv 	   <- calcs ~10 hours based on local MODIS archive, but can be prepared in advance for future days
---grid.csv     	   <- grid_id's features cache
---stmeta.csv              <- stations's features cache
---2022-02-17              <- directory will be contains the submission<timestamp>.csv and inference logging for 2022-02-17
---2022-02-24              <- directory will be contains the submission<timestamp>.csv and inference logging for 2022-02-24
---...
-logs                      <- directory will be contains logging
-models                    <- models weights
-src                       <- python sources
--main.py
--data
--features
--models
--visualization
```

The minimal datasets:

`*` = for training  
`**` = for inference

## Used data

The method uses the following data sources:

- **SNOTEL & CDEC**  
Loading from https://drivendata-public-assets.s3.amazonaws.com/ for inference and from "data/development" for training

- **Copernicus DEM (90 meter resolution)**  
Loading from https://planetarycomputer.microsoft.com/api/stac/v1/collections/cop-dem-glo-90

- **Climate Research Data Package (CRDP) Land Cover Gridded Map (300 meter resolution)**  
Loading from https://drivendata-public-assets.s3.amazonaws.com/ 

- **FAO-UNESCO Global Soil Regions Map**  
Loading from https://drivendata-public-assets.s3.amazonaws.com/ 

- **MODIS Terra MOD10A1 and Aqua MYD10A1**  
Loading from https://modissa.blob.core.windows.net/modis-006

If the local MODIS archive is existed then path can be set as environment variable `MODISPATH`. Only the averaged data are used the precalculated values saving in `data/evaluation/modisclm.csv`

## Running
The solution have common entry point `src/main.py`.

The maindir option is a path to main "snowcast" directory, contains the "data", "models" and "logs" directories.

The scripts calculate the features and saving its values in `data/evaluation` and `data/development` folders. There values are used in the following runs.

### Inference script

```
export MODISPATH=data/modis             #set path to local modis data cache
python3 src/main.py --maindir . --mode oper
```

### Inference perfomance

If inference using the precalculated features 
`data/evaluation/modisclm.csv` (calcs ~10 hours based on local MODIS archive, but can be prepared in advance for future days)
- `data/evaluation/grid.csv`     (calcs one times for grid_id's) 
- `data/evaluation/stmeta.csv`   (calcs one times for SNOTEL & CDEC) 

then the inference tooks about 1 (first submission) - 4 (latest submissions) minutes on Nvidia RTX 2070 and ~6 times slower without GPU.

## Training
```
export MODISPATH=data/modis             #set path to local modis data cache 
python3 src/main.py --maindir . --mode finalize --implicit 1 --relativer 0 --embedding 1 --individual 0
python3 src/main.py --maindir . --mode finalize --implicit 1 --relativer 0 --embedding 0 --individual 0
python3 src/main.py --maindir . --mode finalize --implicit 0 --relativer 0 --embedding 1 --individual 1
python3 src/main.py --maindir . --mode finalize --implicit 0 --relativer 0 --embedding 0 --individual 1
python3 src/main.py --maindir . --mode finalize --implicit 1 --relativer 0 --embedding 1 --individual 0 --modelsize B
python3 src/main.py --maindir . --mode finalize --implicit 1 --relativer 0 --embedding 0 --individual 0 --modelsize B
python3 src/main.py --maindir . --mode finalize --implicit 0 --relativer 0 --embedding 1 --individual 1 --modelsize B
python3 src/main.py --maindir . --mode finalize --implicit 0 --relativer 0 --embedding 0 --individual 1 --modelsize B
```

### Training perfomance
If training using the precalculated features then one training tooks about 15 hours on Nvidia RTX 2070. About 120 hours in total.

### Testing and generation graphs for report
```
export MODISPATH=data/modis             #set path to local modis data cache 
python3 src/main.py --maindir . --mode train --implicit 1 --relativer 0 --embedding 1 --individual 0
python3 src/main.py --maindir . --mode train --implicit 1 --relativer 0 --embedding 0 --individual 0
python3 src/main.py --maindir . --mode train --implicit 0 --relativer 0 --embedding 1 --individual 1
python3 src/main.py --maindir . --mode train --implicit 0 --relativer 0 --embedding 0 --individual 1
python3 src/main.py --maindir . --mode train --implicit 1 --relativer 0 --embedding 1 --individual 0 --modelsize B
python3 src/main.py --maindir . --mode train --implicit 1 --relativer 0 --embedding 0 --individual 0 --modelsize B
python3 src/main.py --maindir . --mode train --implicit 0 --relativer 0 --embedding 1 --individual 1 --modelsize B
python3 src/main.py --maindir . --mode train --implicit 0 --relativer 0 --embedding 0 --individual 1 --modelsize B
python3 src/main.py --maindir . --mode test
```

## Changelog
All changes do not affect to data loading, preparation and submission process.
1. Added visualization for report
2. Improved logging: save `ground_measures_features.csv` to `evaluation\2022-MM-DD` folder

See also `CHANGELOG.md`

