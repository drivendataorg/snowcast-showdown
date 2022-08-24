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
import urllib.request
import io
import numpy as np
import geojson
import pickle
import time

tic = time.perf_counter()

Identifiers = sys.argv[1]
SubmissionID = sys.argv[2]
if ',' in Identifiers:
    ModelIdentifiers = Identifiers.split(',')
else:
    ModelIdentifiers = [Identifiers]

#### Get Data ####
print('Getting Data')

# Download the updated ground measures data
with urllib.request.urlopen('https://drivendata-public-assets.s3.amazonaws.com/ground_measures_features.csv') as response:
   html = response.read().decode("utf-8")
buf = io.StringIO(html)
num_lines = sum(1 for line in buf)  # Number of lines in file

# Read the submission dates
buf = io.StringIO(html)
tline = buf.readline()
header = tline.split(',')
Dates_submit_all = []
for element in header:
    if element[0:2] == '20':
        Dates_submit_all.append(datetime.strptime(element.strip(), '%Y-%m-%d'))

# Only use those in the past
num_times = 0
for Date in Dates_submit_all:
    if Date < datetime.today():
        num_times = num_times + 1

# Read the data
tline = buf.readline()
i = 0
GroundSWE = np.empty([num_lines-1,num_times])
GroundSWE[:] = np.nan
while not (tline == ''):
    fields = tline.split(',')
    for d in range(num_times):
## 2022-06-30 PDB: For the 6/30 submission, the previous code didn't correctly read the data
        # if not (fields[d+1] == ''):
        if not (fields[d+1] == '') and len(fields[d+1]) > 1:
            GroundSWE[i,d] = float(fields[d+1])
    i = i+1
    tline = buf.readline()


# Get IDs of all features
with open('Data/grid_cells.geojson') as f:
    gj = geojson.load(f)
features = gj['features']
cell_ids = []
for feature in features:
    cell_ids.append(feature['properties']['cell_id'])

# Define variables for later
NSites_all = len(cell_ids)
NDates = len(GroundSWE[0, :])
NSites_ground = len(GroundSWE[:, 1])
Dates_submit = Dates_submit_all[0:NDates]

#### Run ML Models ####

print('Running MLR Models')
OutputSWE = np.empty([NSites_all, NDates, len(ModelIdentifiers)])
e = 0

# Loop through the given model identifiers in the ensemble
for ModelIdentifier in ModelIdentifiers:

    # Load the Model Weights
    file = open(r'Models/' + ModelIdentifier + '.dat', 'rb')
    md = pickle.load(file)
    file.close()
    GroundModels = md['GroundModels']
    Models = md['Models']

    # Fill in missing ground measures data
    GroundSWE_filled = np.empty(GroundSWE.shape)
    GroundSWE_filled[:] = 0
    for i in range(NSites_ground):
        a = GroundModels[i]['a']
        weights = GroundModels[i]['weights']
        locs = GroundModels[i]['locs']
        bias = GroundModels[i]['bias']
        if len(GroundModels[i]['a']) > 0:
            for d in range(NDates):
                if not np.nansum(weights * ~np.isnan(GroundSWE[locs, d])) == 0:
                    GroundSWE_filled[i, d] = np.nansum((a * GroundSWE[locs, d]) * weights) / np.nansum(
                        weights * ~np.isnan(GroundSWE[locs, d])) * bias
                else:
                    GroundSWE_filled[i, d] = 0
        else:
            GroundSWE_filled[i, :] = np.nan

    # Use the historical observed values to better constrain filled values
    relax_factor = 0.05
    bias_prev = np.empty(NSites_ground)
    bias_prev[:] = 1
    for d in range(NDates):
        bias = GroundSWE[:, d] / (GroundSWE_filled[:, d] + 1E-5)
        bias[GroundSWE_filled[:, d] == 0] = np.nan
        bias[bias > 5] = 5
        nanlocs = np.isnan(bias)
        bias[nanlocs] = bias_prev[nanlocs]
        GroundSWE_filled[:, d] = GroundSWE_filled[:, d] * bias
        bias_prev = (bias + relax_factor) / (1 + relax_factor)

    # Run the MLR model for each location
    AllSWE_ML = np.empty([NSites_all, NDates])
    AllSWE_ML[:] = 0
    for i in range(NSites_all):
        # Do not run if model doesn't exist
        if not (Models[i]['mdl'] == []):
            x = np.transpose(GroundSWE_filled[Models[i]['locs'], :])
            y = Models[i]['mdl'].predict(x)
            bias = Models[i]['bias']
            y = np.transpose(y) + bias['c']
            y[y < 0] = 0
            AllSWE_ML[i, :] = bias['a'] * y ** bias['b']

            # Boot jumps that are outside the allowable range (3x the 1st and 99th percentiles of observed SWE changes)
            for c in range(NDates):
                if c > 0:
                    diff = AllSWE_ML[i, c] - AllSWE_ML[i, c - 1]
                    if diff < Models[i]['minmax'][0] or diff > Models[i]['minmax'][1]:
                        AllSWE_ML[i, c] = AllSWE_ML[i, c - 1]

    OutputSWE[:, :, e] = AllSWE_ML
    e = e + 1

OutputSWE = np.mean(OutputSWE, axis=2)

#### Generate the Submission File ####
print('Generating the submission file')

# Create output directory if it does not exist
if not os.path.exists('Output CSV'):
    os.makedirs('Output CSV')

# Write the output file
f = open('Output CSV/' + SubmissionID + '.csv', 'w')
f.write(',2022-01-13,2022-01-20,2022-01-27,2022-02-03,2022-02-10,2022-02-17,2022-02-24,2022-03-03,2022-03-10,2022-03-17,2022-03-24,2022-03-31,2022-04-07,2022-04-14,2022-04-21,2022-04-28,2022-05-05,2022-05-12,2022-05-19,2022-05-26,2022-06-02,2022-06-09,2022-06-16,2022-06-23,2022-06-30\n')
for i in range(NSites_all):
    f.write(cell_ids[i])
    for d in range(NDates):
        f.write(',{:.2f}'.format(OutputSWE[i, d]))
    f.write('\n')
f.close()

toc = time.perf_counter()
print(f"Process Done. Elapsed time is {toc - tic:0.2f} seconds")
