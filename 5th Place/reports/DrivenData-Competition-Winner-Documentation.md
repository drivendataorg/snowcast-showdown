# III. Model documentation and writeup

### Who are you (mini-bio) and what do you do professionally? If you are on a team, please complete this block for each member of the team.

I am a Machine Learning Engineer at Uplight Inc., a software company that works with energy utilities to engage customers in new ways for the more efficient use of energy. I received a Masters in Analytics from Georgia Institute of Technology. I enjoy learning about software engineering, machine learning, and physics.

### What motivated you to compete in this challenge?

I am always looking for ways to apply machine learning to interesting and impactful problems. Growing up in Denver, CO I have been fascinated by the mountains my whole life and observed first-hand mountain snow conditions while skiing and hiking in the mountains. Only recently have I realized the importance of mountain snowpack to the sustainability of water resources in the Western United States. This challenge allowed me to practice my data engineering and machine learning skills, learn about satellite and weather data, and create a solution that could help water resource managers make decisions that positively impact water scarcity in the Western United States.

### High level summary of your approach: what did you do and why?

I took a Data-Centric approach of trying to incorporate as many useful features as possible in a gradient boosting model. Because of this, my approach was more heavily reliant on feature engineering. I used the Modis NDSI data, several values from the HRRR climate data, the DEM (Digital Elevation Model), and the ground measures.

Since gradient boosting naturally works with tabular data I used the mean and in some cases the variance of pixel values from the Modis data and the DEM over an entire grid cell. If you eliminate Modis data for a grid cell on days with high cloud cover (recommended) the Modis data becomes sparse, so the Modis features I created used a rolling average of the mean pixel values, one 5 day rolling average and one 15 day. Modis derived features were most important according to my feature importance analysis.

I found the DEM very helpful. Just using the mean and variance of elevation for a grid cell was a useful feature. I also created a feature that I called southern gradient that took the difference between north-south pixels and represents southern exposure for a grid cell, with the idea that snow on South facing slopes melts faster in the Northern hemisphere.

Geo-spatial and time based features were important, I created a feature I called “snow season day” that just counted the days from the beginning of the snow season, November 1, until the last day, June 30. I also just fed in the raw lat and lon as features, I tried fuzzing a little and it may have helped with generalization, but very minimally in my experiments.

The way I incorporated the ground measures was to use the relative SWE, compared to historical average for a ground measure station. Then for each grid cell I took the 15 nearest neighbors relative SWE. That feature reduced the RMSE a bit more.

The usefulness of the HRRR climate data was a little more perplexing to me. I used a three day rolling average for different values (temperature, WEASD - snow water accumulation, snow depth, pressure, precipitation, etc). In the absence of some of the other features the HRRR data provided value but with all the other features the model remained flat (RMSE didn’t improve). I included it for robustness, there was a period last month where the Aqua Modis dataset was unavailable for over half a month.

I used an ensemble of three different gradient boosting implementations, LightGBM, XGBoost, and Catboost. LightGBM performed the best on its own, and it was the fastest to train. You always here about XGBoost being good for data science competitions but I came away very impressed with LightGBM.

It was a fun and interesting competition. I actually like that there was a data engineering component, unlike most Kaggle competitions. I think that gave me a chance to compete because I feel I am pretty strong with data engineering. Hope that helps, feel free to ask any follow up questions.

### Do you have any useful charts, graphs, or visualizations from the process?

I produced many useful charts, graphs, and visualizations which can be found in the appendix of my model report (`reports/Final Snocast Model Report.pdf`).

### Copy and paste the 3 most impactful parts of your code and explain what each does and how it helped your model.

**Neighbor Relative SWE Feature**

The neighbor relative SWE (NRSWE) feature is meant to represent the relative amount of SWE for the neighboring ground measurements of a grid cell, compared to historical values for the snow season period. The following steps show how the NRSWE feature is calculated:

Calculate 14 periods based on the snow season day value to get snow season period (Table 5).

Calculate the historical mean and standard deviation or each ground measurement/snow season period pair.

Use the historical mean and standard deviation to calculate the relative SWE, representing the Z-score (Equation 1) of each ground measurement.

For each grid cell find the 15 nearest neighbors and the distance (Equation 2)  to each neighbor based on latitude and longitude in the ground measurement dataset.

Multiply the weighted inverse distance (Equation 3) for each of the 15 neighbors by each neighbor’s relative SWE value (Equation 4).

The NRSWE feature captures the simple idea that when nearby ground measurement stations are measuring greater SWE than normal, it is likely that the location itself will have a greater SWE than normal. The effects of the NRSWE feature are highly localized by geography

**Modis Snow Cover Daily Terra and Aqua Features**

Several of the most important features were created from the Modis Satellite Snow Cover Daily product. One of the valuable aspects of the Modis NDSI data is it is captured daily from two different sets of satellites. The high-frequency of data capture for Modis allows for the creation of rolling averages, which smooth inherent noise in the measurements, within a time window that is close enough to the prediction date to be relevant to SWE prediction. Since the resolution for the Modis Snow Cover data is 500 meters, there are typically 4-5 relevant measurements for each 1 km^2 grid cell and these were averaged for each grid cell. Next, rolling averages for the Modis Snow Cover Daily data (Terra and Aqua) were compiled with a 5-day and 15-day window from the SWE measurement date, creating the following features: NDSI_Snow_Cover_Terra_5_day, NDSI_Snow_Cover_Aqua_5_day, NDSI_Snow_Cover_Terra_15_day, NDSI_Snow_Cover_Aqua_5_day.

We experimented with different windows for the rolling averages (3, 5, 7, 15, 30) and in the end chose 5 and 15 because that minimized the loss in the models. With hindsight, a 7 day window may be preferable to the 5 day window for near real-time prediction because the Modis data is often available with a 2 day lag and with a 7 day window it is less important if the rolling average is missing values corresponding to the final day prior to SWE prediction.

**Elevation Features**

All of the data sources used to create features were localized using a latitude and longitude value (point measures) or a polygon based on latitude and longitude values (grid cells). The elevation features captured from the DEM demonstrate the differences in acquiring data for point measures and grid cells. The DEM captures elevation measurements at a 30 meter resolution. 

The elevation_m feature represents the elevation in meters. For a point measure the elevation value can be directly queried from the DEM for the 30 meter grid corresponding to that point. To create the elevation_m feature for a 1 km^2 grid cell requires calculating the mean of all elevation values in the grid cell. The elevation_var_m represents the variance of elevation values for the grid cell in the DEM. The elevation_var_m is not applicable to a point measurement. 

The elevation gradient features were also calculated from the DEM (Fig. 2).  The elevation gradients were captured along two axes, east-west and south-north. For each axes the 1st discrete difference14 was calculated for each 30 meter elevation value in the 1 km^2 grid cell. The mean over all of these discrete differences represents the gradient along that axis for the grid cell. For the south_elev_grad feature a positive value represents terrain with a greater Southern exposure than Northern exposure. Similarly, the east_elev_grad feature has a positive value for terrain with a greater Eastern exposure than Western Exposure. Since the elevation gradient features are also applicable to point measures we added and subtracted an epsilon of 0.001 to the latitude and longitude of the point measures to create a grid cell on which to calculate the gradient features.


### Please provide the machine specs and time you used to run your model.

I used Google Colab for my compute resources. I did not require a GPU.

Train duration: ~ 1-2 hours

Inference duration: ~ 15-30 min.

### Anything we should watch out for or be aware of in using your model (e.g. code quirks, memory requirements, numerical stability issues, etc.)?

The model is sensitive to the recency of the Modis data. There were several weeks where Modis data lagged a day or two behind and the model likely produced less accurate predictions on those weeks, especially during the melt season (May, June, July).

### Did you use any tools for data preparation or exploratory data analysis that aren’t listed in your code submission?

N/A

### How did you evaluate performance of the model other than the provided metric, if at all?

We only used temporal stratification to separate the observations and relied on the features of the model to distinguish between samples with different geography, elevation, or snow conditions. Observations were divided into a Train dataset that contained all train labels and ground measures from 2013 to 2019 and a Test dataset that contained labels and ground measures from 2020 and 2021.

We ran 5-fold cross-validation on the Train dataset to tune the hyperparameters of the GBM models and perform feature selection. The metric used to evaluate model performance was the root mean squared error (RMSE). After selecting features and optimal hyperparameters we performed SWE prediction on the Test dataset to evaluate the model’s ability to generalize to unseen data.

For all three GBM implementations the majority of the reduction in RMSE came in the first 100 iterations of training, but each model continued to improve RMSE on the train and test dataset for many subsequent iterations as shown in the learning curves. The LightGBM model achieved the best performance on the Test dataset, followed by the XGBoost model and then the Catboost model. The results in Table 3 demonstrate that the ensemble of GBM models generalized better than any individual model and capitalized on the relative strength of each GBM implementation in certain regions.

### What are some other things you tried that didn’t necessarily make it into the final workflow (quick overview)?

We also thoroughly considered and explored the Sentinel 1 Terrain Corrected Data and the Landsat 8 Collection 2 Level-2 Data before ultimately deciding not to use them in the final model.

Synthetic aperture radar (SAR) measurements from the Sentinel 1 can capture measurements day and night, independent of clouds or atmospheric conditions. The ability of SAR to capture data when Modis and other optical sensor reflectance data is obfuscated could provide valuable information to the model. However, one of the major downsides to SAR data is that terrain has a strong effect on the measurements. The different incident angles of measurement based on satellite location at the time of measurement combined with the mountainous terrain of the train/test locations results in inconsistent measurements. We attempted correcting for incidence angle and normalizing the data for particular locations, but the model performed better without including the Sentinel 1 SAR data.

The data from Landsat 8 Collection 2 Level-2 is promising because it offers a similar product to Modis but with greater resolution Modis (Landsat 30 m to Modis 500 m). The downside of the Landsat data is that it is collected less frequently than the Modis data. Since our modeling method of choice (GBM) requires tabular data inputs we relied on averaging NDSI values within a grid cell, thus losing the advantage of greater resolution. The Landsat data may be useful in models that can take advantage of the greater resolution, like convolutional neural networks. In the end, the Landsat data was captured too infrequently and considered redundant with the Modis data, so we did not include it in the final model.

### If you were to continue working on this problem for the next year, what methods or techniques might you try in order to build on your work so far? Are there other fields or features you felt would have been very helpful to have?

Given more time to work on the challenge of SWE prediction in the Western United States there are several data sources and methods that could improve the results. The training labels were derived from a combination of ground-based snow telemetry (ie. SNOTEL and CDEC sites) and airborne measurements (ie. ASO data). During training it would be useful to know which labels fall into each respective category, even if that information is not available during prediction time. If there are subtle differences in the measurements that impact model performance then it might inform model architecture. 

As described above, with the satellite based imaging data there is a trade off between the higher time frequency, lower spatial resolution of Modis and the lower time frequency, higher spatial resolution of Landsat. In the end, the greater frequency of Modis was more valuable to our model than the greater resolution of Landsat. However, one compelling idea for future work is to combine datasets with greater resolution (Landsat, Sentinel, DEM) with a modeling approach that could take advantage of the greater resolution, like a convolutional neural network. This could allow for more granular SWE predictions with higher resolution than 1 km^2 which may be more accurate if the data can support it.

Another idea for future work is to implement features that capture temporal changes in time-sensitive data, rather than only performing rolling averages. One example is that atmospheric pressure is highly correlated with elevation (higher elevations have lower pressure). So the addition of the feature representing atmospheric pressure (`PRES_3_day`) in our model did not add much information that wasn’t already present in the `elevation_m` feature. However, a feature that represents change in atmospheric pressure over some period of time (i.e. 3-day window) might add relevant information to the model about weather patterns that lead to SWE, and this information should not be as highly correlated with elevation.

A final thought, SWE is a cumulative measure. Modeling changes in SWE throughout the snow season may shed light on the dynamics of SWE accumulation and the conditions that give rise to SWE. It would also be interesting to connect SWE prediction to metrics about SWE runoff and the effects on river conditions and water availability beyond just the snow season.