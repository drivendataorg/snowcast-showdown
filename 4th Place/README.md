# Solution - Snowcast Showdown

Username: leigh-plt

## Summary
- For using code you must download data or datasets, build docker images.
- Input data just normalized on run to range -1..1 for most of data (min-max : -10..10)
- Models trained from scratch with custom architecture.
- Final result will be mean result of 4-fold Cross-Validation models.

## Data used for evaluation:
- [MYD10A1](https://nsidc.org/data/MYD10A1) - data access location: `https://modissa.blob.core.windows.net/modis-006`
- [HRRR](https://rapidrefresh.noaa.gov/hrrr/) - data access location: `https://noaahrrr.blob.core.windows.net/hrrr`
- [Copernicus DEM (90 meter resolution)](https://object.cloud.sdsc.edu/v1/AUTH_opentopography/www/metadata/Copernicus_metadata.pdf) - data access location: `https://planetarycomputer.microsoft.com/api/stac/v1/collections/cop-dem-glo-90`
- [FAO-UNESCO Global Soil Regions Map](https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/use/?cid=nrcs142p2_054013) - data access location: `s3://drivendata-public-assets/soil_regions_map.tar.gz`
- Information about sun daylight duration - Calculus based on date and latitude. Formulas from https://gml.noaa.gov/grad/solcalc/calcdetails.html.

# Setup

1. Install the prerequisities
     - Docker
     - Conda with env: `train/environment.yaml`
     - Build docker image "model:base" by path `images/model/Dockerfile`
     - Build airflow `docker-compose.yaml` for the automated inference process
     - Download data
        - HRRR and MODIS manually or with airflow scheduler
          OR
        - train/inference datasets (See #3)

2. For predictions used docker image, build files located in folder images.

  See file `images/model/environment.yaml` with list of libraries.

  For training read `README.md` in folder train.

Data folder must contained downloaded files HHHR and MODIS or full evaluation dataset.

! MODIS data after 2022/03/30 obtained from Terra spacecraft and renamed as MYD10A1 to simplify execution scripts.

```
└── data
    ├── evaluation_dataset.nc                  <- must containing constant field soil and dem
    ├── hrrr
    │   ├── ...
    │   └── hrrr.20220210
    │       └── conus
    │           ├── hrrr.t00z.wrfsfcf00.grib2
    │           └── hrrr.t12z.wrfsfcf00.grib2
    └── modis
        └── MYD10A1
            └── ...
```

3. Data loading:
  [Training dataset](https://drive.google.com/file/d/1byzZadHONRHZZ0E9kQksP_ZOhhHmABNt/view?usp=sharing) from google drive.

  After 03/30/22 [Aqua Safe Mode Alert](https://lpdaac.usgs.gov/news/aqua-safe-mode-alert/) as source of modis data use Terra spacecraft.
  
  [Full evaluation dataset](https://drive.google.com/file/d/1c-fH88e4m9MQRUBfibs7mrAfN7_OQBWc/view?usp=sharing) from google drive.

# Hardware
### System
Trained with
  memory:         16GiB System memory
  processor:      AMD Ryzen 5 5600X 6-Core Processor

Train time: ~2-3h

The inference was run on AWS EC2 t3.large

Inference time: 15-20 mins with updates dataset for each date.

# Run training
  See `train/README.md`

# Run inference
  For inference build docker image "model:base" by path "images/model/Dockerfile".

  For evaluation date run command:
    `docker run -v inference/data:/data model:base python /data/estimate.py -d 2022-02-10T20:00:00.000000+00:00 --format_file /data/submission_format.csv`
    
    ! format_file will be overridden and headers must contain "date" for evaluation.
