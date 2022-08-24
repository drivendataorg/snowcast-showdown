import cv2
from datetime import datetime, timedelta
import geojson
from geotiff import GeoTiff  
from models import Model0
import netCDF4
import numpy as np
import pandas as pd
import os
from os import path
from scipy import interpolate
from scipy.io import loadmat, savemat
import torch
import wget
import tarfile
import data

cosd = lambda x: np.cos(np.radians(x))

jdays = list(range(0,183+21,7))+list(range(329-21,364,7)) #load Julian days
def getconstfeatures(workdir, uregions, awsurl, print=print):
    datadir = path.join(workdir,'..')
    print(f"getconstfeatures: datadir={datadir} list={os.listdir(datadir)}")    
    file = path.join(workdir,'grid_cells.geojson')
    print(f"Loading {file}")
    with open(file) as f:
        grid0 = geojson.load(f)
    grid0 = pd.DataFrame([{'cell_id':g['properties']['cell_id'], 'region':g['properties']['region'], 
                          'corners': np.array(g['geometry']['coordinates'])} for g in grid0['features']]).set_index('cell_id')
    file = path.join(workdir,'ground_measures_metadata.csv')
    print(f"Loading {file}")
    stmeta0 = pd.read_csv(file).set_index('station_id')
    
    stmetafile = path.join(workdir,'stmeta.csv')
    gridfile = path.join(workdir,'grid.csv')
    read = path.isfile(stmetafile) and path.isfile(gridfile)
    if read:
        print(f'Loading stmeta from {stmetafile} and grid from {gridfile}')
        stmeta = pd.read_csv(stmetafile).set_index('station_id')
        grid = pd.read_csv(gridfile).set_index('cell_id') 
        noex = set(stmeta0.index).difference(set(stmeta.index)).union(set(grid0.index).difference(set(grid.index)))
        if len(noex) > 0:
            print('unvalid stmeta / grid for {noex}')
            read = False
        else:
            lonr = 1.5
            lon1 = np.floor(min(grid['longitude'].values.min(),stmeta['longitude'].values.min())/lonr-1.)*lonr
            lon2 = np.ceil(max(grid['longitude'].values.max(),stmeta['longitude'].values.max())/lonr+1.)*lonr
            latr = 1.
            lat1 = np.floor(min(grid['latitude'].values.min(),stmeta['latitude'].values.min())/latr-1.)*latr
            lat2 = np.ceil(max(grid['latitude'].values.max(),stmeta['latitude'].values.max())/latr+1.)*latr   
    if not read:        
        print('Creating stmeta and grid')
        grid = grid0
        stmeta = stmeta0
        gll = np.vstack(grid['corners'].values)
        grid['latitude'] = gll[:,:,1].mean(1)
        grid['longitude'] = gll[:,:,0].mean(1)
        
        lonr = 1.5; latr = 1.
        lon1 = np.floor(min(gll[:,:,0].min(),stmeta['longitude'].values.min())/lonr-1.)*lonr
        lon2 = np.ceil(max(gll[:,:,0].max(),stmeta['longitude'].values.max())/lonr+1.)*lonr
        lat1 = np.floor(min(gll[:,:,1].min(),stmeta['latitude'].values.min())/latr-1.)*latr
        lat2 = np.ceil(max(gll[:,:,1].max(),stmeta['latitude'].values.max())/latr+1.)*latr
        for lab in uregions:
            grid[lab] = np.array([grid['region'][k]==lab for k in range(grid.shape[0])]).astype(np.float32)
            stmeta[lab] = np.zeros(stmeta.shape[0])
            
        for lab in ['CDEC', 'SNOTEL']:
            stmeta[lab] = np.array([stmeta.index[k][:len(lab)]==lab for k in range(stmeta.shape[0])]).astype(np.float32)
            grid[lab] = np.zeros(grid.shape[0])
                
        rgauss = 2.0
        def getaver (lon,lat,elev,r):
            ry = r/(111.*(lat[1]-lat[0]))
            rx = r/(111.*(lon[1]-lon[0])*cosd((lat1+lat2)*0.5))
            av = elev.copy()
            cv2.GaussianBlur(elev, (2*int(rgauss*rx)+1, 2*int(rgauss*ry)+1), rx, av, ry)
            f = interpolate.interp2d(lon, lat, av, kind='linear')
            return lambda lons, lats: np.array([f(lons[k], lats[k])[0] for k in range(lons.shape[0])])
        
        demfile = f"dem_N{lat1}_{lat2}_W{-lon1}_{-lon2}.mat"
        fname = path.join(datadir, demfile)
        if not path.isfile(fname):
            print('Creating DEM features')
            dem = data.getdem(lat1,lat2,lon1,lon2,dir=path.join(datadir,'dem'), matfile=fname)
        else:
            print(f'Loading {demfile}')
            dem = loadmat(fname)
        demlon = dem.pop('lon').squeeze()
        demlat = dem.pop('lat').squeeze()
        print('Calculation DEM features')
        for key in dem:
            if key[:2] != '__':
                elev = dem[key]
                if key == 'elev':
                    rads = [3, 10, 30, 100]
                    f = getaver(demlon,demlat,elev,1.)
                    grid['elevation_m'] = f(grid['longitude'], grid['latitude'])
                    for r in rads:
                        f_av = getaver(demlon,demlat,elev,r)
                        name = 'elevation_'+str(r)
                        for d in [stmeta, grid]:
                            d[name] = f_av(d['longitude'], d['latitude']) - d['elevation_m']
                else:
                    rads1 = [1, 3, 10, 30]
                    for r in rads1:
                        f_av = getaver(demlon,demlat,elev,r)
                        name = key+str(r)
                        for d in [stmeta, grid]:
                            d[name] = f_av(d['longitude'], d['latitude'])
        ev = getaver(demlon,demlat,dem['elev'],1.)(stmeta['longitude'], stmeta['latitude'])
        print(f"dem elevation/stmeta elevation = {ev/stmeta['elevation_m']}")
        del demlon,demlat,dem
        
        print('Loading GLOBCOVER')  
        for d in [stmeta, grid]:
            for key in [key for key in d.keys() if key[:9]=='GLOBCOVER']:
                d.pop(key)
        
        ncname = path.join(datadir,'C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc')
        if not path.isfile(ncname):
            arch = 'land_cover_map.tar.gz'
            fname = path.join(datadir,arch)
            if not path.isfile(fname):
                print('Downloading '+arch)
                wget.download(awsurl+arch, out=fname)
            tar = tarfile.open(fname, "r:gz").extractall(datadir)
            # ncname = path.join(datadir, tar.getmembers()[0].get_info()['name'])
            os.remove(fname)
        print(f'Loading GLOBCOVER from {ncname}')
        nc = netCDF4.Dataset(ncname)
        lon = np.array(nc.variables['lon'][:])
        lat = np.array(nc.variables['lat'][:])
        ok = ((lat>=lat1)&(lat<=lat2)).nonzero()[0]
        ilat0 = ok[0]; ilat1 = ok[-1]+1
        ok = ((lon>=lon1)&(lon<=lon2)).nonzero()[0]
        ilon0 = ok[0]; ilon1 = ok[-1]+1
        arr = np.array(nc.variables['lccs_class'][0,ilat0:ilat1,ilon0:ilon1])
        lon = lon[ilon0:ilon1]
        lat = lat[ilat0:ilat1]
        nc.close()
        
        printvalstat = lambda arr: print ({t: (arr==t).sum()/arr.size*100. for t in np.unique(arr.reshape(-1))})
        printvalstat (arr)
        arr[(arr>=10) & (arr<30)] = 30
        arr[arr==110] = 100; arr[arr==120] = 100
        arr[(arr>130)&(arr<160)] = 130
        arr[arr==72] = 70; arr[arr==71] = 70
        arr[arr==201] = 200
        types = [30,70,90,100,130,200,210,220]
        printvalstat (arr)
        gstep=1./360.
        # rads = [1, 3, 10, 30]
        rads = [3]
        print('Calculation GLOBCOVER features')
        def calcfeatures(arr,types,gstep,prefix):
            for t in types:
                eq = (arr==t).astype(np.float32)
                for r in rads:
                    ry = r/(111.*gstep)
                    rx = r/(111.*gstep*cosd((lat1+lat2)*0.5))
                    av = eq.copy()
                    cv2.GaussianBlur(eq, (2*int(rgauss*rx)+1, 2*int(rgauss*ry)+1), rx, av, ry)
                    for d in [stmeta, grid]:
                        ilon = ((d['longitude'].values-lon1)/(lon2-lon1)*arr.shape[1]).astype(np.int64)
                        ilat = ((lat2-d['latitude'].values)/(lat2-lat1)*arr.shape[0]).astype(np.int64)
                        d[prefix+str(t)+'_'+str(r)] = np.array([av[ilat[i]:ilat[i]+2,ilon[i]:ilon[i]+2].mean() for i in range(ilon.shape[0])])
            del eq,av
        calcfeatures(arr,types,gstep,'GLOBCOVER')
        del arr
        
        print('Loading SOIL')
        for d in [stmeta, grid]:
            for key in [key for key in d.keys() if key[:4]=='SOIL']:
                d.pop(key)
        tiffile = 'global_soil_regions_geoTIFF/so2015v2.tif'
        tifname = path.join(datadir,tiffile)
        if not path.isfile(tifname):
            arch = 'soil_regions_map.tar.gz'
            fname = path.join(datadir,arch)
            if not path.isfile(fname):
                print('Downloading '+arch)
                wget.download(awsurl+arch, out=fname)
            tar = tarfile.open(fname, "r:gz").extract('./'+tiffile, datadir)
            os.remove(fname)
        
        print(f'Loading SOIL from {tifname}')
        arr = np.array(GeoTiff(tifname).read_box([(lon1,lat1),(lon2,lat2)]))
        printvalstat (arr)
        # types = [7,21,50,54,64,74,75,81,83,92]
        arr[arr>10] = np.floor(arr[arr>10]/10)*10
        arr[arr==5] = 7; arr[arr==6] = 7
        printvalstat (arr)
        types = [7,20,50,60,70,80,90]
        # types = np.unique(arr.reshape(-1))
        gstep = 1./30.
        # rads = [3, 10, 30]
        rads = [10]
        print('Calculation SOIL features')
        calcfeatures(arr,types,gstep,'SOIL')
        del arr
        
        # clm = 'ba'
        # print('Loading '+clm)
        # badir = path.join(datadir, clm+'-nc')
        # if not path.isdir(badir):
        #     arch = 'burned_areas_occurrence_map.tar.gz'
        #     fname = path.join(datadir,arch)
        #     if not path.isfile(fname):
        #         print('Downloading '+arch)
        #         wget.download(awsurl+arch, out=fname)
        #     tar = tarfile.open(fname, "r:gz").extractall(datadir)
        #     os.remove(fname)            
        # rads = [10, 30]
        # for jd in jdays:
        #     if all([clm+str(r)+'_'+str(jd) in grid for r in rads]):
        #         continue
        #     tday = (datetime(2001,1,1)+timedelta(days=jd)).strftime('%m%d')
        #     file = path.join(badir,'ESACCI-LC-L4-'+clm+'-Cond-500m-P13Y7D-2000'+tday+'-v2.0.nc')
        #     print(f'Loading {clm} {tday} from {file}')
        #     nc = netCDF4.Dataset(file)
        #     lon = np.array(nc.variables['lon'][:])
        #     lat = np.array(nc.variables['lat'][:])
        #     ok = ((lat>=lat1)&(lat<=lat2)).nonzero()[0]
        #     ilat0 = ok[0]; ilat1 = ok[-1]+1
        #     ok = ((lon>=lon1)&(lon<=lon2)).nonzero()[0]
        #     ilon0 = ok[0]; ilon1 = ok[-1]+1
        #     v = np.array(nc.variables[clm.lower()+'_occ'][ilat0:ilat1,ilon0:ilon1]).astype(np.float32)
        #     lon = lon[ilon0:ilon1]
        #     lat = lat[ilat0:ilat1]
        #     for r in rads:
        #         f = getaver(lon, lat, v, r)
        #         for d in [stmeta, grid]:
        #             d[clm+str(r)+'_'+str(jd)] = f (d['longitude'], d['latitude'])
        #     nc.close()
            
        stmeta = stmeta.copy()
        grid = grid.copy()
        
        print('Saving stmeta to {stmetafile} and grid to {gridfile}')
        stmeta.to_csv(stmetafile)
        grid.to_csv(gridfile)
        
        print({key: grid[key].mean() for key in grid.keys() if key not in ['region', 'corners']})
        print({key: stmeta[key].mean() for key in stmeta.keys() if key not in ['name','state']})
        
    print('Interpolate regions tags')
    dtype = torch.float32
    x = {'xlo': stmeta['longitude'].values, 'xla': stmeta['latitude'].values,
         'ylo':   grid['longitude'].values, 'yla':   grid['latitude'].values}
    x = {key: torch.tensor(x[key], dtype=dtype)[None] for key in x}
    for lab in ['CDEC', 'SNOTEL']:
        x['xval'] = torch.tensor(stmeta[lab].values, dtype=dtype)[None,:,None]
        grid[lab] = Model0(x)[0,:,0].detach().numpy()
    x = {key: x[('y' if key[0]=='x' else 'x')+key[1:]] for key in x if key[1:] in ['lo','la']}
    for lab in uregions:
        x['xval'] = torch.tensor(grid[lab].values, dtype=dtype)[None,:,None]
        stmeta[lab] = Model0(x)[0,:,0].detach().numpy()
    constfeatures = ['CDEC', 'elevation_m']
    rads = [100, 30, 10, 3]
    # rads = [100, 10]
    # rads = [30, 10, 3]
    constfeatures += ['elevation_'+str(r) for r in rads]
    for d in [stmeta, grid]:
        for r,r2 in zip(rads[1:],rads[:-1]):
            d['elevation_'+str(r2)] -= d['elevation_'+str(r)]
    # rads = [1, 3, 10, 30]
    rads = [1, 3, 30]
    for key in ['south', 'east']:
        constfeatures += [key+str(r) for r in rads]
        for r,r2 in zip(rads[1:],rads[:-1]):
            for d in [stmeta, grid]:
                # print([key,r2,np.abs(d[key+str(r2)]).mean(), r,np.abs(d[key+str(r)]).mean(),np.abs(d[key+str(r2)] - d[key+str(r)]).mean()])
                d[key+str(r2)] -= d[key+str(r)]
    rads = [1, 3, 10, 30]
    for key in ['aspect']:
        constfeatures += [key+str(r) for r in rads]    
        for r,r2 in zip(rads[1:],rads[:-1]):
            for d in [stmeta, grid]:
                d[key+str(r2)] -= d[key+str(r)]
    # constfeatures += [key for key in grid if key[:9]=='GLOBCOVER' and key[-2:] in ['_1','10']] # and key[9:12] != '220'
    # constfeatures += [key for key in grid if key[:4]=='SOIL' and key[-2:] in ['_3','30']]
    constfeatures += [key for key in grid if key[:9]=='GLOBCOVER' and key[-2:] in ['_3']]
    constfeatures += [key for key in grid if key[:4]=='SOIL' and key[-2:] in ['10']]
    # constfeatures += [key for key in grid if (key[:9]=='GLOBCOVER') or (key[:4]=='SOIL')]
    print(f"constfeatures : {constfeatures}")
    return stmeta,grid,constfeatures