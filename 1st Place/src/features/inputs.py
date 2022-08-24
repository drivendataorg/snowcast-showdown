import gc
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from os import path
import torch

import features

def getinputs(workdir, rmode, modelsdir, withpred=True, nmonths=1, maindevice=torch.device('cpu'),
              awsurl='https://drivendata-public-assets.s3.amazonaws.com/', print=print):
    uregions = ['central rockies', 'sierras']
    stmeta,grid,constfeatures = features.getconstfeatures(workdir, uregions, awsurl = awsurl, print=print)
    
    file = path.join(workdir,'submission_format.csv')
    print(f"Loading submission format from {file}")
    sub = pd.read_csv(file)
    for tday in features.getdays(sub):
        sub[tday] = np.full(sub.shape[0], np.nan)
    sub = sub.rename({sub.columns[0]: 'cell_id'}, axis=1).set_index('cell_id')
    gridswe_test  = sub.join (grid, on='cell_id')
    stswe = {}; gridswe = {}
        
    if rmode == 'oper':
        file = awsurl+'ground_measures_features.csv'
        print(f"Downloading {file}")
        stswe_test = pd.read_csv(file)
        file = path.join(workdir,'ground_measures_features.csv')
        stswe_test.to_csv(file)
        # print(f"Loading {file}")
        # stswe_test = pd.read_csv(file)
    elif rmode == 'test':
        file = path.join(workdir,'ground_measures_test_features.csv')
        print(f"Loading {file}")
        stswe_test = pd.read_csv(file)
    else:
        sttestfile = path.join(workdir,'ground_measures_test_features.csv')
        sttrainfile = path.join(workdir,'ground_measures_train_features.csv')
        gridtrainfile = path.join(workdir,'train_labels.csv')
        print(f"Loading {sttestfile} {sttrainfile} {gridtrainfile}")
        stswe_test = pd.read_csv(sttestfile)
        stswe_train = pd.read_csv(sttrainfile)
        stswe_train = stswe_train.rename({stswe_train.columns[0]: 'station_id'}, axis=1).set_index('station_id')
        gridswe_train = pd.read_csv(gridtrainfile).set_index('cell_id')
        
        stswe_train = stswe_train.join (stmeta, on='station_id')
        gridswe_train = gridswe_train.join (grid, on='cell_id')    
        
        stswe['train'] = stswe_train
        gridswe['train'] = gridswe_train
    stswe_test = stswe_test.rename({stswe_test.columns[0]: 'station_id'}, axis=1).set_index('station_id').join (stmeta, on='station_id')
    
    stswe['test'] = stswe_test
    gridswe['test'] = gridswe_test
    
    days = {mode: features.getdays(stswe[mode]) for mode in stswe}
    dates = {mode: [datetime.strptime(tday,'%Y-%m-%d') for tday in days[mode]] for mode in stswe}
    
    stswe, gridswe, nstations  = features.getembindex (stswe, gridswe, days, modelsdir=modelsdir, print=print)
    stswe, gridswe, rsfeatures = features.getmodisfeatures (workdir, stswe, gridswe, dates, rmode, print=print)
    
    if rmode != 'oper':
        file = path.join(workdir,'labels_2020_2021.csv')
        print(f"Loading {file}")
        trg = pd.read_csv(file).set_index('cell_id')
        for key in trg:
            if key in gridswe['test']:
                gridswe['test'].pop(key)
        gridswe['test'] = gridswe['test'].join (trg, on='cell_id')
        if rmode[:8] == 'finalize':
            for d in [stswe, gridswe]:
                for key in d['test']:
                    if key not in d['train']:
                        d['train'][key] = d['test'][key]
                d.pop('test')
            days['train'] = days['train']+days.pop('test')
            dates['train'] = dates['train']+dates.pop('test')
        if rmode[:5] == 'train':
            # days['train'] = days['train'][:-5]
            # dates['train'] = dates['train'][:-5]
            for d in [stswe, gridswe]:
                for key in d['test']:
                    if key not in d['train']:
                        d['train'][key] = d['test'][key]
            days['train'] = days['train']+days['test'][:-31]
            dates['train'] = dates['train']+dates['test'][:-31]
            days['test'] = days['test'][-31:]
            dates['test'] = dates['test'][-31:]
            print(f"test: {days['test']}")
            print(f"train: {days['train']}")
                
    constfeatures = uregions + list(sorted(constfeatures)) + ['isemb']
    rsfeatures = list(sorted(rsfeatures))
    gc.collect()
    inputs, arglist, days, dates = features.getdatadict(rmode, stswe, gridswe, constfeatures, rsfeatures, maindevice,
                                   withpred=withpred, nmonths=nmonths, print=print)    
    return inputs, arglist, uregions, stswe, gridswe, days, dates, nstations
