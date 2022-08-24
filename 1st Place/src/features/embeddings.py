import numpy as np
from os import path
import pandas as pd
from scipy import spatial

def getembindex(stswe, gridswe, days, modelsdir=path.join('..','models'), print=print):
    fname = path.join(modelsdir,'eindex.csv')
    if path.isfile(fname):
        eindex = pd.read_csv(fname)
    elif 'train' in stswe:
        eindex = [stswe[mode].index for mode in stswe]+[
                gridswe['train'].index[np.vstack([np.isfinite(gridswe['train'][d]) for d in days['train']]).sum(0) > 10]]
        eindex = np.unique(np.hstack(eindex))
        eindex = pd.DataFrame({'emb': np.arange(eindex.shape[0]+1), 'index': np.hstack(['None']+[eindex])})
        for lab in ['longitude', 'latitude']:
            eindex[lab] = np.full(eindex.shape[0],np.nan)
        for mode in stswe:
            _,i1,j1 = np.intersect1d(eindex['index'],stswe[mode].index,return_indices=True)
            _,i2,j2 = np.intersect1d(eindex['index'],gridswe[mode].index,return_indices=True)
            for lab in ['longitude', 'latitude']:
                eindex[lab].values[i1] = stswe[mode][lab].values[j1]
                eindex[lab].values[i2] = gridswe[mode][lab].values[j2]
        eindex.set_index('index').to_csv(fname)
    else:
        print ("Cannot load "+fname)
        
    nstations = eindex.shape[0]
    for mode in stswe:
        for d in [stswe, gridswe]:
            tree = spatial.KDTree(np.vstack((eindex['longitude'],eindex['latitude'])).T)
            dis,ind = tree.query(np.vstack((d[mode]['longitude'], d[mode]['latitude'])).T)
            ok = np.nonzero(dis<0.02)[0]
            ind = ind[ok]
            d[mode]['emb1'] = np.zeros(d[mode].shape[0],dtype=np.int64)
            d[mode]['emb1'].values[ok] = eindex['emb'].values[ind]
    eindex.pop('latitude')
    eindex.pop('longitude')            
    for mode in stswe:
        if 'emb' in stswe[mode]:
            stswe[mode].pop('emb')
            gridswe[mode].pop('emb')
        stswe[mode] = stswe[mode].join(eindex.rename({'index': 'station_id'}, axis=1).set_index('station_id'))
        gridswe[mode] = gridswe[mode].join(eindex.rename({'index': 'cell_id'}, axis=1).set_index('cell_id'))
        for d in [gridswe[mode], stswe[mode]]:
            d['femb'] = np.isnan(d['emb'])
            d['emb'].values[d['femb']] = d['emb1'].values[d['femb']]
            d['emb'] = d['emb'].astype(np.int64)
            d['isemb'] = (d['emb']==0).astype(np.float32)
        
    # eindex = pd.read_csv(fname)
    # for d in [stswe, gridswe]:
    #     for mode in d:
    #         vi = (((d[mode]['emb'].values!=0)&(d[mode]['emb1'].values==0))|((d[mode]['emb1'].values!=0)&(d[mode]['emb'].values==0))).nonzero()[0]
    #         e = d[mode]['emb'][vi]
    #         e1 = d[mode]['emb1'][vi]
    #         print({lab: np.vstack([eindex[lab].values[e],eindex[lab].values[e1]]).T for lab in ['longitude', 'latitude']})
    isst = np.array([len(i)<20 for i in eindex['index']])
    isst[0] = False
    for mode in gridswe:
        if mode == 'test':
            gridswe[mode]['fstation'] = isst[gridswe[mode]['emb1']] & gridswe[mode]['femb']
            # for tday in days[mode]:
            #     g = gridswe[mode][tday].values[gridswe[mode]['fstation']]
            #     ok = np.isfinite(g)
            #     if ok.sum():
            #         print(f"{mode} {tday} {g[ok]}")
        else:
            gridswe[mode]['fstation'] = np.full(gridswe[mode].shape[0], False)
    # for mode in gridswe:
    #     vi = isst[gridswe[mode]['emb1']] & gridswe[mode]['femb']
    #     es = stswe[mode]['emb'].values.copy()
    #     es[es==0] = -1
    #     _,i,j = np.intersect1d(gridswe[mode]['emb1'].values*vi,es,return_indices=True)
    #     for tday in days[mode]:
    #         g = gridswe[mode][tday].values[i]
    #         s = stswe[mode][tday].values[j]
    #         ok = np.isfinite(g+s) # & (s==0)
    #         if ok.sum():
    #             print(f"{mode} {tday} {np.abs(g[ok]-s[ok]).mean()} {np.vstack((g[ok],s[ok])).T}")
                # print(f"{mode} {tday} {np.vstack((gridswe[mode]['longitude'].values[i][ok],gridswe[mode]['latitude'].values[i][ok],stswe[mode]['longitude'].values[j][ok],stswe[mode]['latitude'].values[j][ok],g[ok],s[ok])).T}")
    return stswe, gridswe, nstations