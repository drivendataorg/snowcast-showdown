from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from os import path
from scipy.io import loadmat, savemat

try:
    from data.modis import getinterpolated,getmeanfields,interpfields
except:
    print("Cannot use data.modis. Install requirements")

def initmodisnsdi (stswe,gridswe):
    modisnsdi = {}
    for mode in stswe:
        modisnsdi[mode] = pd.DataFrame( {'index':np.hstack((stswe[mode].index,gridswe[mode].index)),
                                         'latitude': np.hstack((stswe[mode]['latitude'].values,gridswe[mode]['latitude'].values)), 
                                         'longitude': np.hstack((stswe[mode]['longitude'].values,gridswe[mode]['longitude'].values))}  )
    return pd.concat([modisnsdi[m] for m in stswe]).drop_duplicates("index")

def getmodisfeatures (workdir, stswe, gridswe, dates, rmode, print=print):
    # # lat,lon = np.meshgrid(np.arange(lat1,lat2,0.1),np.arange(lon1,lon2,0.1))
    # # r = getinterpolated('MYD10A1', lat.reshape(-1), lon.reshape(-1), dates['train'][-1], [1])
    # # plt.contourf(lon,lat,r['NDSI1'].reshape(lon.shape))
    fmt = '%Y-%m-%d'
    print ('Loading modis')
    today = datetime.now()
    reqdates = np.unique(np.hstack([dates[m] for m in dates]))
    reqdays = [date.strftime(fmt) for date in reqdates]
    if rmode == 'oper':
        loaddates = ['2013-01-01', '2013-01-08', '2013-01-15', '2013-01-22', '2013-01-29', '2013-02-05', '2013-02-12', '2013-02-19', '2013-02-26', '2013-03-05', 
                     '2013-03-12', '2013-03-19', '2013-03-26', '2013-04-02', '2013-04-09', '2013-04-16', '2013-04-23', '2013-04-30', '2013-05-07', '2013-05-14', 
                     '2013-05-21', '2013-05-28', '2013-06-04', '2013-06-11', '2013-06-18', '2013-06-25', '2013-12-03', '2013-12-10', '2013-12-17', '2013-12-24', 
                     '2013-12-31', '2014-01-07', '2014-01-14', '2014-01-21', '2014-01-28', '2014-02-04', '2014-02-11', '2014-02-18', '2014-02-25', '2014-03-04', 
                     '2014-03-11', '2014-03-18', '2014-03-25', '2014-04-01', '2014-04-08', '2014-04-15', '2014-04-22', '2014-04-29', '2014-05-06', '2014-05-13', 
                     '2014-05-20', '2014-05-27', '2014-06-03', '2014-06-10', '2014-06-17', '2014-06-24', '2014-12-02', '2014-12-09', '2014-12-16', '2014-12-23', 
                     '2014-12-30', '2015-01-06', '2015-01-13', '2015-01-20', '2015-01-27', '2015-02-03', '2015-02-10', '2015-02-17', '2015-02-24', '2015-03-03', 
                     '2015-03-10', '2015-03-17', '2015-03-24', '2015-03-31', '2015-04-07', '2015-04-14', '2015-04-21', '2015-04-28', '2015-05-05', '2015-05-12', 
                     '2015-05-19', '2015-05-26', '2015-06-02', '2015-06-09', '2015-06-16', '2015-06-23', '2015-06-30', '2015-12-01', '2015-12-08', '2015-12-15', 
                     '2015-12-22', '2015-12-29', '2016-01-05', '2016-01-12', '2016-01-19', '2016-01-26', '2016-02-02', '2016-02-09', '2016-02-16', '2016-02-23', 
                     '2016-03-01', '2016-03-08', '2016-03-15', '2016-03-22', '2016-03-29', '2016-04-05', '2016-04-12', '2016-04-19', '2016-04-26', '2016-05-03', 
                     '2016-05-10', '2016-05-17', '2016-05-24', '2016-05-31', '2016-06-07', '2016-06-14', '2016-06-21', '2016-06-28', '2016-12-06', '2016-12-13', 
                     '2016-12-20', '2016-12-27', '2017-01-03', '2017-01-10', '2017-01-17', '2017-01-24', '2017-01-31', '2017-02-07', '2017-02-14', '2017-02-21', 
                     '2017-02-28', '2017-03-07', '2017-03-14', '2017-03-21', '2017-03-28', '2017-04-04', '2017-04-11', '2017-04-18', '2017-04-25', '2017-05-02', 
                     '2017-05-09', '2017-05-16', '2017-05-23', '2017-05-30', '2017-06-06', '2017-06-13', '2017-06-20', '2017-06-27', '2017-12-05', '2017-12-12', 
                     '2017-12-19', '2017-12-26', '2018-01-02', '2018-01-09', '2018-01-16', '2018-01-23', '2018-01-30', '2018-02-06', '2018-02-13', '2018-02-20', 
                     '2018-02-27', '2018-03-06', '2018-03-13', '2018-03-20', '2018-03-27', '2018-04-03', '2018-04-10', '2018-04-17', '2018-04-24', '2018-05-01', 
                     '2018-05-08', '2018-05-15', '2018-05-22', '2018-05-29', '2018-06-05', '2018-06-12', '2018-06-19', '2018-06-26', '2018-12-04', '2018-12-11', 
                     '2018-12-18', '2018-12-25', '2019-01-01', '2019-01-08', '2019-01-15', '2019-01-22', '2019-01-29', '2019-02-05', '2019-02-12', '2019-02-19', 
                     '2019-02-26', '2019-03-05', '2019-03-12', '2019-03-19', '2019-03-26', '2019-04-02', '2019-04-09', '2019-04-16', '2019-04-23', '2019-04-30', 
                     '2019-05-07', '2019-05-14', '2019-05-21', '2019-05-28', '2019-06-04', '2019-06-11', '2019-06-18', '2019-06-25', '2019-12-03', '2019-12-10', 
                     '2019-12-17', '2019-12-24', '2019-12-31', '2020-01-07', '2020-01-14', '2020-01-21', '2020-01-28', '2020-02-04', '2020-02-11', '2020-02-18', 
                     '2020-02-25', '2020-03-03', '2020-03-10', '2020-03-17', '2020-03-24', '2020-03-31', '2020-04-07', '2020-04-14', '2020-04-21', '2020-04-28', 
                     '2020-05-05', '2020-05-12', '2020-05-19', '2020-05-26', '2020-06-02', '2020-06-09', '2020-06-16', '2020-06-23', '2020-06-30', '2020-12-01', 
                     '2020-12-08', '2020-12-15', '2020-12-22', '2020-12-29', '2021-01-05', '2021-01-12', '2021-01-19', '2021-01-26', '2021-02-02', '2021-02-09', 
                     '2021-02-16', '2021-02-23', '2021-03-02', '2021-03-09', '2021-03-16', '2021-03-23', '2021-03-30', '2021-04-06', '2021-04-13', '2021-04-20',
                     '2021-04-27', '2021-05-04', '2021-05-11', '2021-05-18', '2021-05-25', '2021-06-01', '2021-06-08', '2021-06-15', '2021-06-22', '2021-06-29']
        loaddates = [datetime.strptime(d,fmt) for d in loaddates]
    else:
        loaddates = []
    loaddates = np.unique(np.hstack([datetime(2013,1,1)-timedelta(days=7*dt) for dt in range(1,1200)]+[reqdates,loaddates]))
    loaddates = [d for d in loaddates if (d<=today or d in reqdates) and d>=datetime(2000,2,24) and (d.month<=7 or d.month>=11)]    
    loaddays = [date.strftime(fmt) for date in loaddates]
    rads = modisrads = [1, 3, 10, 30]
    modisnsdi0 = initmodisnsdi (stswe,gridswe)
    index = set(modisnsdi0['index'].values)
    if False:
    # if rmode in ['oper','test']: 
    # if rmode in ['oper']: 
        modistestfile = path.join(workdir, 'modistest.csv')
        if path.isfile(modistestfile):
            modisnsdi = pd.read_csv(modistestfile)
        else:
            modisfieldsfile = path.join(workdir, 'modisfields.mat')
            if path.isfile(modisfieldsfile):
                modisfields = loadmat(modisfieldsfile)
            else:
                modisfields = getmeanfields(reqdates, reqdays, loaddates)
                savemat(modisfieldsfile, modisfields, do_compression=True)
            modisnsdi = interpfields(modisfields, modisnsdi0, reqdates, reqdays, modisrads)
            mds = 'MNDSI_Snow_Cover'
            for r,r2 in zip(modisrads[1:],modisrads[:-1]):
                mdsr = mds+str(r); mdsr2 = mds+str(r2)
                for tday in reqdays:
                    if tday+mdsr in modisnsdi:
                        modisnsdi[tday+mdsr2] = np.nan_to_num(modisnsdi[tday+mdsr2]-modisnsdi[tday+mdsr],0.)
            modisnsdi.set_index('index').to_csv(modistestfile)
    else:
        modisclmfile = path.join(workdir, 'modisclm.csv')
        modisnsdi = {}
        if path.isfile(modisclmfile):            
            modisnsdi = pd.read_csv(modisclmfile)
            noex = index.difference(set(modisnsdi['index'].values))
            if len(noex) > 0:
                print(f"Not found modis features for {noex}")
                modisnsdi = {}
            else:
                print(f"Loaded {modisclmfile}")
        mds = 'NDSI_Snow_Cover'
        lastday = loaddays[-1]
        # if not ((lastday+mds+'1' in modisnsdi) or (lastday+'M'+mds+'10' in modisnsdi)):
        if not (lastday+'M'+mds+'10' in modisnsdi):
            modisfile = path.join(workdir, 'modis.csv')
            if path.isfile(modisfile):
                modisnsdi = pd.read_csv(modisfile)
                noex = index.difference(set(modisnsdi['index'].values))
                if len(noex) > 0:
                    print(f"Not found modis features for {noex}")
                    modisnsdi = modisnsdi0
                else:
                    print(f"Loaded {modisfile}")
            else:
                modisnsdi = modisnsdi0
            lastday = [d for d in loaddates if d<=today][-1].strftime(fmt)
            if not ((lastday+'O'+mds+'1' in modisnsdi) or (lastday+'Y'+mds+'1' in modisnsdi)):                
                def getmodis(modisnsdi, tday, rads=modisrads, loady=True):
                    # print ('Loading '+tday)
                    msg = {}
                    for product in ['MOD10A1', 'MYD10A1'][:1+loady]:
                        if tday+product[1]+mds+str(rads[0]) not in modisnsdi:
                            rs = getinterpolated(product, modisnsdi['latitude'].values, modisnsdi['longitude'].values, datetime.strptime(tday,fmt), rads)
                            for key in rs:
                                modisnsdi[tday+product[1]+key] = rs[key]
                                msg[product[1]+key] = np.isfinite(rs[key]).sum()
                    if len(msg) > 0:
                        print (f"{tday}: {msg}")
                for date in loaddates:
                    try:
                        getmodis (modisnsdi, date.strftime(fmt), loady=date>=datetime(2002,7,4))
                    except:
                        print(f"Cannot loading modis for {date}")
                modisnsdi.set_index('index').to_csv(modisfile)        
            print ('Calculating modis features')
            # {tday: np.isnan(modisnsdi[tday+'YNDSI1']).sum() for tday in days['test']}
            # for mds in ['ONDSI_Snow_Cover', 'ONDSI', 'YNDSI_Snow_Cover', 'YNDSI']:
            # for mds in ['NDSI_Snow_Cover', 'NDSI']:
            for mds in ['NDSI_Snow_Cover']:
                if mds[:4] == 'NDSI':            
                    for tday in loaddays:
                        for r in modisrads:
                            mdsr = mds+str(r)
                            if tday+'Y'+mdsr in modisnsdi:
                                modisnsdi[tday+mdsr] = np.nanmean([modisnsdi[tday+'Y'+mdsr].values,modisnsdi[tday+'O'+mdsr].values],0) \
                                    if tday+'O'+mdsr in modisnsdi else modisnsdi[tday+'Y'+mdsr]
                            else:
                                if tday+'O'+mdsr in modisnsdi:
                                    modisnsdi[tday+mdsr] = modisnsdi[tday+'O'+mdsr]
                        for r2 in modisrads:
                            for product in 'OY':
                                if tday+product+mds+str(r2) in modisnsdi:
                                    modisnsdi.pop(tday+product+mds+str(r2))        
                for r,r2 in zip(modisrads[1:],modisrads[:-1]):
                    mdsr = mds+str(r); mdsr2 = mds+str(r2)
                    for tday in loaddays:
                        if tday+mdsr in modisnsdi:
                            modisnsdi[tday+mdsr2] = np.nan_to_num(modisnsdi[tday+mdsr2]-modisnsdi[tday+mdsr],0.)
                for r in modisrads:
                    mdsr = mds+str(r)
                    for date,tday in zip (reqdates, reqdays):
                        if tday+'M'+mdsr not in modisnsdi:
                            averd = [tday1 for date1,tday1 in zip (loaddates, loaddays) if np.abs(((date-date1).days+180)%365-180)<=14 and
                                     np.abs((date-date1).days)>180 and tday1+mdsr in modisnsdi]
                            #     averd = [tday1 for date1,tday1 in zip (loaddates, loaddays) if np.abs(((date-date1).days+180)%365-180)<=14 and
                            #          date1<date-timedelta(days=180) and tday1+mdsr in modisnsdi]
                            if len(averd) > 10:
                                modisnsdi[tday+'M'+mdsr] = np.nanmean([modisnsdi[tday1+mdsr].values for tday1 in averd], 0)
                                if tday+mdsr in modisnsdi:
                                    bad = np.isnan(modisnsdi[tday+mdsr])
                                    modisnsdi[tday+mdsr][bad] = modisnsdi[tday+'M'+mdsr][bad]
                                elif date <= today:
                                    modisnsdi[tday+mdsr] = modisnsdi[tday+'M'+mdsr]
                # for r,r2 in zip(modisrads[1:],modisrads[:-1]):
                #     mdsr = mds+str(r); mdsr2 = mds+str(r2)
                #     for tday in reqdays:
                #         if tday+mdsr in modisnsdi:
                #             modisnsdi[tday+mdsr2] = np.nan_to_num(modisnsdi[tday+mdsr2]-modisnsdi[tday+mdsr],0.)
                #         if tday+'M'+mdsr in modisnsdi:
                #             modisnsdi[tday+'M'+mdsr2] = np.nan_to_num(modisnsdi[tday+'M'+mdsr2]-modisnsdi[tday+'M'+mdsr],0.)
            for key in list(modisnsdi.keys()):
                if key[:4].isnumeric():
                    if key[:10] not in reqdays:
                        # print(key)
                        modisnsdi.pop(key)
                    elif key[10] != 'M':
                        if key[:10]+'M'+key[10:] in modisnsdi:
                            modisnsdi[key] -= modisnsdi[key[:10]+'M'+key[10:]]
                        else:
                            modisnsdi.pop(key)        
            print ('Saving modis features')
            modisnsdi.set_index('index').to_csv(modisclmfile)
    modisnsdi.pop('latitude')
    modisnsdi.pop('longitude')
    # rads = [1, 10, 30]
    rads = modisrads
    rsfeatures = []
    # for mds in ['NDSI_Snow_Cover', 'NDSI']:
    for mds in ['NDSI_Snow_Cover']:
        # rsfeatures += [mds+str(r) for r in [10,30]]
        rsfeatures += ['M'+mds+str(r) for r in rads]
        # rsfeatures += [mds+str(r)+'_m' for r in rads]
        # for r in rads:
        #     mdsr = mds+str(r)
        #     for date,tday in zip (reqdates, reqdays):
        #         averd = []
        #         for dt in range(4):
        #             tday1 = (date-timedelta(days=7*dt)).strftime(fmt)+mdsr
        #             if tday1 in modisnsdi:
        #                 averd.append(tday1)
        #         modisnsdi[tday+mdsr+'_m'] = np.nanmean([modisnsdi[tday1].values for tday1 in averd], 0)
    for mode in stswe:
        stswe[mode] = stswe[mode].join(modisnsdi.rename({'index': 'station_id'}, axis=1).set_index('station_id'))
        gridswe[mode] = gridswe[mode].join(modisnsdi.rename({'index': 'cell_id'}, axis=1).set_index('cell_id'))
    del modisnsdi
    print(f"Modis loaded. rsfeatures : {rsfeatures}")
    return stswe, gridswe, rsfeatures