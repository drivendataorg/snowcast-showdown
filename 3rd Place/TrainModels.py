#******************************************************************************
#  Copyright (c) 2022, Patrick Broxton <broxtopd@arizona.edu>
# 
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#******************************************************************************

import sys
import os
from datetime import datetime
import numpy as np
import geojson
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import pickle
import time

tic = time.perf_counter()

# Parameters for Filling Missing Ground Measures Data

MinNSamples_ground = 50     # Number of ground samples required to consider the ground station
MinRSquare_ground = 0.95    # Minimum RSquare for station to be considered for filling missing ground data
MaxStations_ground = 75     # Number of nearest potential stations for filling missing ground data
MinStations_ground = 20     # Minimum number of stations to be considered for filling of
                            # missing ground measure data
GroundSyncExp = 10          # Exponent applied to RSquare value to figure out station weights
                            # for filling of missing ground measure data

# Parameters for Multilinear Regression
MinNSamples = 50            # Number of ground samples required to use actual ground data for
                            # MLR model training (instead of UA data)
MinRSquare = 0.95           # Minimum RSquare for station to be considered in MLR Algorithm
MinStations = 5             # Minimum number of stations to be considered in MLR Algorithm
MaxStations = 75            # Number of nearest potential stations picked up by MLR Algorithm
ValidRangeMult = 10         # Multiplier applied to range of dSWE to flag for bad values when running the MLR

# Parameters for Bias Correction
MinBiasSamples = 2          # Minimum number of samples to perform bias correction using ground data
CFBounds = ((0.5, 0.75), (1.5, 1.25))   # Bounds of the a*x^b relationship (min-a,min-b)(max-a,max-b) for bias correction


# objective function (for bias correction)
def objective(x, a, b):
    return a * x ** b


# Read in command line arguments
ValidationYears_input = sys.argv[1].split(',')
ModelIdentifier = sys.argv[2]
ValidationYears = []
for ValidationYear in ValidationYears_input:
    if ValidationYear.isnumeric():
        ValidationYears.append(int(ValidationYear))

#### Load Training Data and UA SWE Data ####

print('Loading Training Data and UA SWE Data')

# Load Ground Measures Dates
f = open('Data/ground_measures_data.csv', 'r')
line = f.readline()
header = line.split(',')
Dates_ground = []
for element in header:
    if element[0:2] == '20':
        Dates_ground.append(datetime.strptime(element.strip(), '%Y-%m-%d'))
f.close()

# Get Latitude and Longitude of Ground Measures Data
f = open('Data/ground_measures_metadata.csv', 'r')
GroundLats = []
GroundLons = []
i = 0
for line in f:
    i = i + 1
    if i > 1:
        fields = line.split(',')
        GroundLats.append(float(fields[3]))
        GroundLons.append(float(fields[4]))
f.close()

# Load Ground Measures Data
with open('Data/ground_measures_data.csv') as f:
    ncols = len(f.readline().split(','))
GroundSWE = np.loadtxt(open('Data/ground_measures_data.csv', 'rb'), delimiter=",", skiprows=1, usecols=range(1, ncols))

# Load Dates for All Locations
f = open('Data/train_label_data.csv', 'r')
line = f.readline()
header = line.split(',')
Dates_all = []
for element in header:
    if element[0:2] == '20':
        Dates_all.append(datetime.strptime(element.strip(), '%Y-%m-%d'))
f.close()

# Get Latitude, Longitude, and IDs of all features
with open('Data/grid_cells.geojson') as f:
    gj = geojson.load(f)
features = gj['features']
AllLats = []
AllLons = []
for feature in features:
    coordinates = feature['geometry']['coordinates']
    AllLons.append((coordinates[0][0][0] + coordinates[0][1][0] + coordinates[0][2][0] + coordinates[0][3][0]) / 4)
    AllLats.append((coordinates[0][0][1] + coordinates[0][1][1] + coordinates[0][2][1] + coordinates[0][3][1]) / 4)

# Load Training Label Data
with open('Data/train_label_data.csv') as f:
    ncols = len(f.readline().split(','))
AllSWE = np.loadtxt(open('Data/train_label_data.csv', 'rb'), delimiter=",", skiprows=1, usecols=range(1, ncols))

# Load UA SWE Data
with open('Data/ua_swe_data.csv') as f:
    ncols = len(f.readline().split(','))
AllUASWE = np.loadtxt(open('Data/ua_swe_data.csv', 'rb'), delimiter=",", skiprows=1, usecols=range(1, ncols))

# Define variables for later
NSites_all = len(AllSWE[:, 1])
NDates_all = len(Dates_all)
NSites_ground = len(GroundSWE[:, 1])

# Syncronize dates for ground measures and other calibration data
GroundSWE_sync = np.empty((NSites_ground, NDates_all))
for i in range(NDates_all):
    loc = -1
    for Date_ground in Dates_ground:
        if Date_ground > Dates_all[i]:
            break
        loc = loc + 1
    GroundSWE_sync[:, i] = GroundSWE[:, loc]
GroundSWE = GroundSWE_sync
Dates_ground = Dates_all

#### Divide Datasets into Training and Testing Portions ####

# Define which water year each sample belongs to for ground dates
years = []
for Date_ground in Dates_ground:
    if Date_ground.month == 12:     # Count December as belonging to the next year
        years.append(Date_ground.year + 1)
    else:
        years.append(Date_ground.year)

# Get training and test locs based on whether the year is a validation year
test_locs = np.empty(len(years), dtype=bool)
test_locs[:] = False
for g in range(len(test_locs)):
    if years[g] in ValidationYears:
        test_locs[g] = True
train_locs = ~test_locs

# Divide each dataset into training and test sets
GroundSWE_train = GroundSWE[:, train_locs]
GroundSWE_test = GroundSWE[:, test_locs]
AllSWE_train = AllSWE[:, train_locs]
AllSWE_test = AllSWE[:, test_locs]
AllUASWE_train = AllUASWE[:, train_locs]

# Define date vectors for each set
Dates_train = []
Dates_test = []
c = 0
for Date_ground in Dates_ground:
    if train_locs[c]:
        Dates_train.append(Date_ground)
    else:
        Dates_test.append(Date_ground)
    c = c + 1

# Other variables for later
NDates_train = len(Dates_train)
NDates_test = len(Dates_test)

#### Fill missing values in the ground data set ####

print('Creating Models to fill missing ground data')

# Loop through all sites to create a model for filling missing data
# Then apply them to training set - we will apply to the test set later
GroundSWE_train_filled = np.empty(GroundSWE_train.shape)
GroundSWE_train_filled[:] = np.nan
GroundModels = []
for i in range(NSites_ground):
    t = GroundSWE_train[i, :]
    t = t + np.random.rand(len(t)) * 1E-5       # Add tiny fluctuations to prevent all zeros

    # Only consider stations with > MinNSamples data
    if np.count_nonzero(~np.isnan(t)) >= MinNSamples_ground:

        # Select MaxStations training sites closest to the target site
        dist = np.empty([NSites_ground])
        for j in range(NSites_ground):
            dist[j] = np.sqrt((GroundLats[i] - GroundLats[j]) ** 2 + (GroundLons[i] - GroundLons[j]) ** 2)
        idx = np.argsort(dist)
        locs = idx[1:MaxStations_ground+1]
        x = np.transpose(GroundSWE_train[locs, :])
        x = x + np.random.rand(x.shape[0], x.shape[1]) * 1E-5

        # Then test the regressions between each site and the target site
        # (and only select stations that have R2 > MinRSquare_ground, or a
        # minimum of MinStations_ground training locations
        rsquares = np.empty(len(locs))
        rsquares[:] = -999
        for j in range(len(locs)):
            x_ = t
            y_ = x[:, j]
            nanlocs = np.isnan(y_) | np.isnan(x_)
            if np.count_nonzero(~nanlocs) >= MinNSamples_ground:
                corr_matrix = np.corrcoef(x_[~nanlocs], y_[~nanlocs])
                rsquares[j] = corr_matrix[0, 1] ** 2
        sorted_rsquares = np.sort(rsquares)[::-1]
        idx = np.argsort(rsquares)[::-1]
        MinStations_g = np.minimum((sorted_rsquares > 0.0).sum(), MinStations_ground)   # In unlikely case that there are not enough stations with non-missing data
        locs = locs[idx[:max(MinStations_g, np.count_nonzero(sorted_rsquares >= MinRSquare_ground))]]
        x = np.transpose(GroundSWE_train[locs, :])
        x = x + np.random.rand(x.shape[0], x.shape[1]) * 1E-5

        # Perform regression between target station and each potential predictor, and save regression slope,
        # its assigned weighting (as a function of the r2 values and the GroundSyncExp parameter)
        a = np.empty(len(locs))
        weights = np.empty(len(locs))
        for j in range(len(locs)):
            x_ = t
            y_ = x[:, j]
            nanlocs = np.isnan(y_) | np.isnan(x_)
            X_ = x_[~nanlocs] - x_[~nanlocs].mean()
            Y_ = y_[~nanlocs] - y_[~nanlocs].mean()
            a[j] = (X_.dot(Y_)) / (X_.dot(X_))
            weights[j] = sorted_rsquares[j] ** GroundSyncExp

        # Fill the missing ground station data in the training set
        filled = np.empty(NDates_train)
        for d in range(NDates_train):
            if np.nansum(weights * ~np.isnan(GroundSWE_train[locs, d])).sum() > 0:
                filled[d] = np.nansum((a * GroundSWE_train[locs, d]) * weights) / np.nansum(
                    weights * ~np.isnan(GroundSWE_train[locs, d]))
            else:       # Unlikely case where we have 0 in the denominator
                if d > 0:
                    filled[d] = filled[d-1]
                else:
                    filled[d] = 0

        # Compute a bias between the actual and modeled values, and apply this bias to the model
        nonanlocs = ~np.isnan(GroundSWE_train[i, :])
        if np.mean(filled[nonanlocs]) > 0:
            bias = np.mean(GroundSWE_train[i, nonanlocs]) / np.mean(filled[nonanlocs])
        else:       # Unlikely case where we have 0 in the denominator
            bias = 0
            
        GroundSWE_train_filled[i, :] = filled * bias

        GroundModels.append({'a': a, 'weights': weights, 'locs': locs, 'bias': bias})
    else:
        # If a ground station is not used, then fill with empty variables
        GroundModels.append({'a': [], 'weights': [], 'locs': [], 'bias': []})

# Use the historical observed values to better constrain filled values
# This algorithm makes the 'filled' data match the observed data where observed data exists,
# and guesses at biases when there is missing data based on what they were before data was missing
relax_factor = 0.05         # Slowly, biases work back toward 1 if multiple days of missing data
bias_prev = np.empty(NSites_ground)
bias_prev[:] = 1
for d in range(NDates_train):
    bias = GroundSWE_train[:, d] / (GroundSWE_train_filled[:, d] + 1E-5)
    bias[GroundSWE_train_filled[:, d] == 0] = np.nan
    bias[bias > 5] = 5
    nanlocs = np.isnan(bias)
    bias[nanlocs] = bias_prev[nanlocs]
    GroundSWE_train_filled[:, d] = GroundSWE_train_filled[:, d] * bias
    bias_prev = (bias + relax_factor) / (1 + relax_factor)

#### Train MLR Models ####

print('Training MLR Models')
Models = []
for i in range(NSites_all):
    # If there are enough samples, use the observed ground data as the
    # target, otherwise use the UA data
    if np.count_nonzero(~np.isnan(AllSWE_train[i, :])) >= MinNSamples:
        t = AllSWE_train[i, :]
    else:
        t = AllUASWE_train[i, :]
    t = t + np.random.rand(len(t)) * 1E-5   # Add tiny fluctuations to prevent all zeros

    diff = t[1:]-t[0:-1]
    minmax = [ValidRangeMult*np.percentile(diff, 1), ValidRangeMult*np.percentile(diff, 99)]

    # Select MaxStations training sites closest to the target site
    dist = np.empty([NSites_ground])
    for j in range(NSites_ground):
        dist[j] = np.sqrt((AllLats[i] - GroundLats[j]) ** 2 + (AllLons[i] - GroundLons[j]) ** 2)
    idx = np.argsort(dist)
    locs = idx[:MaxStations]
    x = np.transpose(GroundSWE_train_filled[locs, :])
    x = x + np.random.rand(x.shape[0], x.shape[1]) * 1E-5

    # Then test the regressions between each site and the target site
    # (and only select stations that have R2 > MinRSquare, or a
    # minimum of MinStations training locations
    rsquares = np.empty(len(locs))
    rsquares[:] = -999
    for j in range(len(locs)):
        if np.count_nonzero(~np.isnan(x[:, j])) >= MinNSamples:
            x_ = t
            y_ = x[:, j]
            nanlocs = np.isnan(y_) | np.isnan(x_)
            corr_matrix = np.corrcoef(x_[~nanlocs], y_[~nanlocs])
            rsquares[j] = corr_matrix[0, 1] ** 2
    sorted_rsquares = np.sort(rsquares)[::-1]
    idx = np.argsort(rsquares)[::-1]
    locs = locs[idx[:max(MinStations, np.count_nonzero(sorted_rsquares >= MinRSquare))]]
    x = np.transpose(GroundSWE_train_filled[locs, :])
    x = x + np.random.rand(x.shape[0], x.shape[1]) * 1E-5

    # Train the MLR Model and apply to the training data
    nanlocs = np.isnan(t)
    mdl = LinearRegression().fit(x[~nanlocs, :], t[~nanlocs])
    if np.max(mdl.coef_) > 10:
        # If targets are all close to zero (i.e. very little to no snow), the model may be 
        # unstable.  In this case, don't make a model prediction.
        mdl = []
        locs = []
        bias = []
        minmax = []

    if not (mdl == []):
        y = mdl.predict(x)
        y[y < 0] = 0

        bias = []
        y = np.transpose(y)     # First, look at snow-free condition and make sure it clamps to zero
        if np.percentile(y, 30) - np.percentile(y, 25) < 0.01:
            c = -np.percentile(y, 30)
        if np.percentile(y, 25) - np.percentile(y, 20) < 0.01:
            c = -np.percentile(y, 25)
        if np.percentile(y, 20) - np.percentile(y, 15) < 0.01:
            c = -np.percentile(y, 20)
        elif np.percentile(y, 15) - np.percentile(y, 10) < 0.01:
            c = -np.percentile(y, 15)
        elif np.percentile(y, 10) - np.percentile(y, 5) < 0.01:
            c = -np.percentile(y, 10)
        elif np.percentile(y, 5) - np.percentile(y, 0) < 0.01:
            c = -np.percentile(y, 5)
        else:
            c = 0

        # Next, if sufficient observed data exist, find the bias for the modeled SWE values
        if np.sum(AllSWE_train[i, :] > 0) >= MinBiasSamples:
            t = AllSWE_train[i, :]
            t[y == np.min(y)] = 0
            t = t + np.random.rand(len(t)) * 1E-5
            y = y + np.random.rand(len(t)) * 1E-5
            nanlocs = np.isnan(t)
            xdata = y[~nanlocs]
            ydata = t[~nanlocs]
            try:
                popt, _ = curve_fit(objective, xdata, ydata, bounds=CFBounds)
                a, b = popt
            except:     # If curve fitting doesn't work, accept no bias correction
                a = 1
                b = 1
            n = np.sum(~nanlocs)
        else:
            a = 1
            b = 1
            n = 0

        bias = {'a': a, 'b': b, 'c': c, 'n': n}

    # Keep track of the MLR model, bias, and which ground stations are used
    Models.append({'mdl': mdl, 'locs': locs, 'bias': bias, 'minmax': minmax})
    
# Save the model data to file
print('Saving Models/' + ModelIdentifier + '.dat')
if not os.path.exists('Models'):
    os.makedirs('Models')
file = open(r'Models/' + ModelIdentifier + '.dat', 'wb')
pickle.dump({'GroundModels': GroundModels, 'Models': Models}, file)
file.close()

#### Run ML Models ####

if not (ValidationYears == []):
    print('Running MLR Models for Validation Period')

    # Fill in missing ground measures data
    GroundSWE_test_filled = np.empty(GroundSWE_test.shape)
    GroundSWE_test_filled[:] = np.nan
    for i in range(NSites_ground):
        a = GroundModels[i]['a']
        weights = GroundModels[i]['weights']
        locs = GroundModels[i]['locs']
        bias = GroundModels[i]['bias']
        if len(GroundModels[i]['a']) > 0:
            for d in range(NDates_test):
                if not np.nansum(weights * ~np.isnan(GroundSWE[locs, d])) == 0:
                    GroundSWE_test_filled[i, d] = np.nansum((a * GroundSWE_test[locs, d]) * weights) / np.nansum(weights * ~np.isnan(GroundSWE[locs, d])) * bias
                else:
                    GroundSWE_test_filled[i, d] = 0
        else:
            GroundSWE_test_filled[i, :] = np.nan

    # Use the historical observed values to better constrain filled values
    relax_factor = 0.05
    bias_prev = np.empty(NSites_ground)
    bias_prev[:] = 1
    for d in range(NDates_test):
        bias = GroundSWE_test[:, d] / (GroundSWE_test_filled[:, d] + 1E-5)
        bias[GroundSWE_test_filled[:, d] == 0] = np.nan
        bias[bias > 5] = 5
        nanlocs = np.isnan(bias)
        bias[nanlocs] = bias_prev[nanlocs]
        GroundSWE_test_filled[:, d] = GroundSWE_test_filled[:, d] * bias
        bias_prev = (bias + relax_factor) / (1 + relax_factor)

    # Run the MLR model for each location
    AllSWE_ML_test = np.empty(AllSWE_test.shape)
    AllSWE_ML_test[:] = 0
    for i in range(NSites_all):
        if not (Models[i]['mdl'] == []):
            x = np.transpose(GroundSWE_test_filled[Models[i]['locs'], :])
            y = Models[i]['mdl'].predict(x)
            bias = Models[i]['bias']
            y = np.transpose(y) + bias['c']
            y[y < 0] = 0
            AllSWE_ML_test[i, :] = bias['a'] * y ** bias['b']

            # Boot jumps that are outside the allowable range (3x the 1st and 99th percentiles of observed SWE changes)
            for c in range(NDates_test):
                if c > 0:
                    diff = AllSWE_ML_test[i, c] - AllSWE_ML_test[i, c-1]
                    if diff < Models[i]['minmax'][0] or diff > Models[i]['minmax'][1]:
                        AllSWE_ML_test[i, c] = AllSWE_ML_test[i, c-1]

    # Print Statistics
    x = AllSWE_test
    y = AllSWE_ML_test
    nanlocs = np.isnan(x) | np.isnan(y)
    RMSE = np.sqrt(np.mean((x[~nanlocs] - y[~nanlocs]) ** 2))
    RSquare = 1 - np.sum((x[~nanlocs] - y[~nanlocs]) ** 2) / np.sum((x[~nanlocs] - np.mean(x[~nanlocs])) ** 2)
    Mean = np.mean(x[~nanlocs])

    print('RSquare: {:.2f}'.format(RSquare))
    print('RMSE: {:.2f}'.format(RMSE) + ' (which is {:.0f}'.format(RMSE/Mean*100) + ' percent of the mean)')

toc = time.perf_counter()
print(f"Process Done. Elapsed time is {toc - tic:0.2f} seconds")