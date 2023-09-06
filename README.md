[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

<img src='https://s3.amazonaws.com/drivendata-public-assets/swe-reclamation1.jpg' width='600'>

# Snowcast Showdown

## Goal of the Competition

Seasonal mountain snowpack is a critical water resource throughout the Western U.S. Snowpack acts as a natural reservoir by storing precipitation throughout the winter months and releasing it as snowmelt when temperatures rise during the spring and summer. This meltwater becomes runoff and serves as a primary freshwater source for major streams, rivers and reservoirs. As a result, snowpack accumulation on high-elevation mountains significantly influences streamflow as well as water storage and allocation for millions of people.

**The goal of the Snowcast Showdown challenge was to estimate snow water equivalent (SWE) in real-time each week for 1km x 1km grid cells across the Western U.S.** Getting better high-resolution SWE estimates for mountain watersheds and headwater catchments will help to improve runoff and water supply forecasts, which in turn will help reservoir operators manage limited water supplies. Improved SWE information will also help water managers respond to extreme weather events such as floods and droughts.

## What's in this Repository

This repository contains code from winning competitors in the [Snowcast Showdown](https://www.drivendata.org/competitions/90/competition-reclamation-snow-water-eval/) DrivenData challenge. Code for all winning solutions are open source under the MIT License.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

### Real-time Prediction

Place |Team or User | Score | Summary of Model
--- | --- | --- | ---
1   | FBykov | 3.8784 | Two layers perceptrons are used to estimate the mean, the spread, and the mapping in latent space. The final result is a mix of 63 inhomogeneous kriging models. The models have 8 different configurations and use different train-validation splitting.
2   | UltimateHydrology | 3.9746 | The solution is based on different SOTA implementations of Gradient Boosting Machine algorithm: XGBoost, LightGBM, and CatBoost, and their ensembles. The top features of 121 features set included ground snow measure data (SNOTEL, CDEC), and remote sensing of snow cover (MODIS Terra MOD10A1). The top 1-4 features include physically-based indirect predictors of SWE: seasonal cumulative sum of solid precipitation, seasonal average values of air temperature and the mean seasonal value of solar radiation.
3   | TeamUArizona | 3.9900 | Multilinear Regression (MLR) models are used to predict SWE based on the provided snow station data. Models are trained to predict either ground measurements of SWE (if there are enough measurements for a particular grid cell), or SWE data taken from a gridded SWE dataset called the University of Arizona (UA) SWE dataset (if they are not). In addition to the MLR models, there is also code to fill missing snow station predictor data and to perform some bias correction of the models if necessary. Ultimately, model predictions are made by averaging an ensemble of MLR models.
4   | leigh.plt | 4.0892 | We used a neural network model with different layer architectures: Fourier neural operator (FNO), convolution layers, embeddings, and linear transformation. This architecture allows us to combine data of different nature: images, historical data and classification labels.
5   | oshbocker | 4.1828 | I used an ensemble of three different gradient boosting implementations, LightGBM, XGBoost, and Catboost. Since gradient boosting naturally works with tabular data, I used the mean and in some cases the variance of pixel values from the Modis data and the DEM over an entire grid cell. I also created a feature representing southern sun exposure and a feature capturing the number of days since the beginning of the snow season. The way I incorporated the ground measures was to use the relative SWE, compared to historical average for a ground measure station. Then for each grid cell I took the 15 nearest neighbors relative SWE.

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Interviews with the winners: ["Meet the Winners of the Snowcast Showdown Competition"](https://drivendata.co/blog/swe-winners)**

### Modeling Report

Everyone who successfully submitted a model for real-time evaluation was invited to submit a [modeling report](https://www.drivendata.org/competitions/86/competition-reclamation-snow-water-dev/page/415/#report) that discusses their solution methodology and explains its performance on historical data. Reports were evaluated on the following criteria:

- **Interpretability**: To what extent can a person understand the outcome of the solution methodology given its inputs?
- **Robustness**: Do solutions provide high performance over a broad range of geographic and climate conditions?
- **Rigor**: To what extent is the report built on sound, sophisticated quantitative analysis and a performant statistical model?
- **Clarity**: How clearly are underlying patterns exposed, communicated, and visualized?

Place | Team or User | Place in Prediction Competition | Link
--- | --- | --- | ---
1   | oshbocker | 5th | [Read the report](https://github.com/drivendataorg/snowcast-showdown/blob/main/5th%20Place/reports/Model%20Report.pdf)
2   | UltimateHydrology |  2nd | [Read the report](https://github.com/drivendataorg/snowcast-showdown/blob/main/2nd%20Place/reports/Model%20report%20UltimateHydrology.pdf)
3   | TeamUArizona | 3rd| [Read the report](https://github.com/drivendataorg/snowcast-showdown/blob/main/3rd%20Place/reports/TeamUArizona_Modeling_Report.pdf)

### Regional Prizes

[Regional prizes](https://www.drivendata.org/competitions/90/competition-reclamation-snow-water-eval/page/533/#regional) are awarded for regional performance in the Sierras and Central Rockies. Submissions were evaluated against regional grid cells in the real-time evaluation period. Final rankings were determined by the scoring metric, RMSE.

#### Sierras Regional Prize

Place | Team or User | Place in Prediction Competition
--- | --- | ---
1   | leigh.plt | 4th
2   | FBykov | 1st
3   | UltimateHydrology | 2nd

#### Central Rockies Regional Prize

Place | Team or User | Place in Prediction Competition
--- | --- | ---
1   | UltimateHydrology | 2nd
2   | oshbocker | 5th
3   | leigh.plt | 4th
