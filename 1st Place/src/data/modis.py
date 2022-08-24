from azure.storage.blob import ContainerClient
import cv2
from datetime import datetime, timedelta
import numpy as np
import os
from pyhdf.SD import SD, SDC
from pyproj import Proj,transform
import re
from scipy.interpolate import interp2d
import wget

class ModisSource:
    modis_account_name = 'modissa'
    modis_container_name = 'modis-006'
    modis_account_url = 'https://' + modis_account_name + '.blob.core.windows.net/'
    modis_blob_root = modis_account_url + modis_container_name + '/'
        
    temp_dir = os.environ['MODISPATH']
    if not os.path.isdir(temp_dir):
        import tempfile
        temp_dir = os.path.join(tempfile.gettempdir(),'modis')
        os.makedirs(temp_dir,exist_ok=True)
    tempfiles = os.listdir(temp_dir)
    
    print(f"data\\modis.py: Modis dir is {temp_dir} contain {len(tempfiles)} files")

    fname = 'sn_bound_10deg.txt'
    fn = os.path.join(temp_dir, fname)
    if not os.path.isfile(fn):
        wget.download(modis_blob_root + fname, fn)
    
    # Load this file into a table, where each row is (v,h,lonmin,lonmax,latmin,latmax)
    modis_tile_extents = np.genfromtxt(fn, skip_header = 7, skip_footer = 3)
    modis_container_client = ContainerClient(account_url=modis_account_url, 
                                             container_name=modis_container_name,
                                             credential=None)
    
def lat_lon_to_modis_tiles(lat,lon):
    ok = (lat >= ModisSource.modis_tile_extents[:, 4]) & (lat <= ModisSource.modis_tile_extents[:, 5]) \
         & (lon >= ModisSource.modis_tile_extents[:, 2]) & (lon <= ModisSource.modis_tile_extents[:, 3])
    return ModisSource.modis_tile_extents[ok.nonzero()[0], 1::-1].astype(np.int64)
def lat_lon_to_modis_tile(lat,lon):
    """
    Get the modis tile indices (h,v) for a given lat/lon    
    https://www.earthdatascience.org/tutorials/convert-modis-tile-to-lat-lon/
    """    
    ok = (lat >= ModisSource.modis_tile_extents[:, 4]) & (lat <= ModisSource.modis_tile_extents[:, 5]) \
         & (lon >= ModisSource.modis_tile_extents[:, 2]) & (lon <= ModisSource.modis_tile_extents[:, 3])
    i = ok.nonzero()[0]
    i = i[1] if len(i) >=3 else i[-1]
    return int(ModisSource.modis_tile_extents[i, 1]),int(ModisSource.modis_tile_extents[i, 0])

def list_blobs_in_folder(container_name,folder_name):
    generator = ModisSource.modis_container_client.list_blobs(name_starts_with=folder_name)
    return [blob.name for blob in generator]
            
def list_hdf_blobs_in_folder(container_name,folder_name):
    files = list_blobs_in_folder(container_name,folder_name)
    return [fn for fn in files if fn.endswith('.hdf')]

ul_regex = re.compile(r'''UpperLeftPointMtrs=\(
                          (?P<upper_left_x>[+-]?\d+\.\d+)
                          ,
                          (?P<upper_left_y>[+-]?\d+\.\d+)
                          \)''', re.VERBOSE)
lr_regex = re.compile(r'''LowerRightMtrs=\(
                          (?P<lower_right_x>[+-]?\d+\.\d+)
                          ,
                          (?P<lower_right_y>[+-]?\d+\.\d+)
                          \)''', re.VERBOSE)

# keys = ['NDSI_Snow_Cover', 'NDSI_Snow_Cover_Basic_QA', 'NDSI']
# keys = ['NDSI_Snow_Cover', 'NDSI']
keys = ['NDSI_Snow_Cover']
qakey = 'NDSI_Snow_Cover_Basic_QA'

def calibrate(hdf, keys=None):
    ga = hdf.attributes()['StructMetadata.0']
    match = ul_regex.search(ga)
    x0 = float(match.group('upper_left_x'))
    y0 = float(match.group('upper_left_y'))
    match = lr_regex.search(ga)
    x1 = float(match.group('lower_right_x'))
    y1 = float(match.group('lower_right_y'))
    out = {}
    for key in hdf.datasets():
        if keys is not None:
            if key not in keys and key != qakey:
                continue
        f = hdf.select(key)
        data = f.get()
        attr = f.attributes()    
        # print(attr)
        data_c = data.astype(np.float32)
        if 'scale_factor' in attr:
            data_c = data_c * float(attr['scale_factor'])            
        if '_FillValue' in attr:
            data_c[data==attr['_FillValue']] = np.nan
        if 'valid_range' in attr:
            valid_range = attr['valid_range']
            data_c = np.minimum(np.maximum(data_c, valid_range[0]), valid_range[1])
        out[key] = data_c
    qa = out[qakey]==4
    out['NDSI_Snow_Cover'][qa] = np.nan
    if qakey not in keys:
        out.pop(qakey)
    return out,x0,x1,y0,y1
    
def getmodis_hv(product, h, v, date, verbose=True):
    # Files are stored according to:
    # http://modissa.blob.core.windows.net/modis-006/[product]/[htile]/[vtile]/[year][day]/filename
    # This is the MODIS surface reflectance product
    folder = product + '/' + '{:0>2d}/{:0>2d}'.format(h,v) + '/' + date.strftime('%Y%j')
    # Find all HDF files from this tile on this day    
    # filename = folder+'/'+product+'.A'+date.strftime('%Y%j')+'.h{:0>2d}v{:0>2d}'.format(h,v)+'.006'+
    filename = folder.replace('/','_')
    filenames = [f for f in ModisSource.tempfiles if f[:len(filename)]==filename]
    if len(filenames) == 0:
        ModisSource.tempfiles = os.listdir(ModisSource.temp_dir)
        filenames = [f for f in ModisSource.tempfiles if f[:len(filename)]==filename]
    ex = False
    if len(filenames) > 0:        
        filename = os.path.join(ModisSource.temp_dir,filenames[0])
        ex = os.path.isfile(filename)
    if not ex:
        filenames = list_hdf_blobs_in_folder(ModisSource.modis_container_name,folder)
        if verbose:
            print('Found {} matching file(s):'.format(len(filenames)))
            for fn in filenames:
                print(fn)    
        if len(filenames) == 0:
            return None
        filename = os.path.join(ModisSource.temp_dir, filenames[0].replace('/','_'))
        print(filename)
        if not os.path.isfile(filename):
            # Download to a temporary file
            wget.download(ModisSource.modis_blob_root + filenames[0], filename)
    return SD(filename, SDC.READ)

def getmodis_lat_lon(product, lat, lon, date, verbose=True):
    h,v = lat_lon_to_modis_tile(lat,lon)
    return getmodis_hv(product, h, v, date, verbose)

rgauss = 2.0
def getaver (lon,lat,lons,lats,elev,ex,r):
    ry = r/(1110./(lat.shape[0]-1))
    mask = (2*int(rgauss*ry)+1, 2*int(rgauss*ry)+1)
    av = np.nan_to_num(cv2.GaussianBlur(elev, mask, ry)/cv2.GaussianBlur(ex, mask, ry),-9999.)
    f = interp2d(lon, lat, av, kind='linear')
    ret = np.array([f(lons[k], lats[k])[0] for k in range(lons.shape[0])])
    ret[ret<0.] = np.nan
    return ret

sinu = Proj('+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')
# wgs84 = Proj("+init=EPSG:4326")
# ih = np.array([[8,4],[8,5],[9,4],[9,5],[10,4]])
ih = np.array([[9,5],[8,4],[8,5],[9,4],[10,4]])
def getinterpolated(product, lat, lon, date, rads, verbose=False, keys=keys, new=True):    
    out = {}
    if keys is not None:
        if new:
            for r in rads:
                out.update({key+str(r): np.full_like(lat, np.nan) for key in keys})
        else:
            out.update({key: np.full_like(lat, np.nan) for key in keys})
    # x, y = transform(wgs84, sinu, lon, lat)
    x, y = sinu(lon, lat)
    ihs = [lat_lon_to_modis_tiles(latk,lonk) for lonk,latk in zip(lon,lat)]
    ih = np.unique(np.concatenate (ihs,0), axis=0)
    # badhv = np.array([[7,5],[8,4],[11,4]])    
    for ifile in range(ih.shape[0]):
        hdf = getmodis_hv(product, ih[ifile][0], ih[ifile][1], date, verbose=False)
        if hdf is None:
            continue
        m,x0,x1,y0,y1 = calibrate(hdf, keys=keys)
        sz = m[list(m.keys())[0]].shape
        ok = (x<=x1) & (y>=y1) & (x>=x0) & (y<=y0) 
        xm = np.linspace(x0,x1,sz[0])
        ym = np.linspace(y0,y1,sz[1])
        xs=x[ok]; ys=y[ok]
        print(f"ih={ih[ifile]} count={ok.sum()}")
        for key in m:
            ex = np.isfinite(m[key]).astype(np.float32)
            fld = np.nan_to_num(m[key],0.)
            for r in rads:
                v = getaver (xm,ym,xs,ys,fld,ex,r)
                if key+str(r) in out:
                    v0 = out[key+str(r)][ok]
                    vi = np.isnan(v)
                    v[vi] = v0[vi]
                out[key+str(r)][ok] = v
    bad = [key for key in out if np.isnan(out[key]).all()]
    for key in bad:
        out.pop(key)
    if verbose:
        print([date, {key: np.isfinite(out[key]).sum() for key in out}])
    return out

# hdf = getmodis_hv('MOD10A1', 8,4,datetime(2003,6,10))
# m,x0,x1,y0,y1 = calibrate(hdf, keys=['NDSI_Snow_Cover', 'NDSI_Snow_Cover_Basic_QA', 'NDSI'])
def getfields(product, date, h, v, out={}, verbose=False, keys=keys):
    hdf = getmodis_hv(product, h, v, date, verbose=verbose)
    if hdf is not None:        
        m,x0,x1,y0,y1 = calibrate(hdf, keys=keys)
        out['corners'] = x0,x1,y0,y1
        for key in m:            
            ex = np.isfinite(m[key]).astype(np.float32)
            fld = np.nan_to_num(m[key],0.)
            if key in out:
                out[key] += fld; out[key+'_c'] += ex
            else:
                out[key]  = fld; out[key+'_c']  = ex
            print(f"{product} {date} {h}_{v} {key} {fld.sum()/ex.sum():.4}")
    return out

def getmeanfields(reqdates, reqdays, loaddates, keys=keys, out = {}):
    reqdays = [d.replace('-','_') for d in reqdays]    
    for ifile in range(ih.shape[0]):
        hvstr = f'f_{ih[ifile][0]}_{ih[ifile][1]}_'
        for date in loaddates:
            read = False
            for date1,tday1 in zip (reqdates, reqdays):
                if np.abs(((date-date1).days+180)%365-180)<=14:
                    read = True
                    break
            if read:
                ret = [getfields(product, date, ih[ifile][0], ih[ifile][1], keys=keys) 
                       for product in ['MOD10A1', 'MYD10A1'][:1+(date>=datetime(2002,7,4))]]
                ret = {key: np.nanmean([r[key] for r in ret if len(r)>0],0) for key in keys+['corners']}
                for date1,tday1 in zip (reqdates, reqdays):
                    if np.abs(((date-date1).days+180)%365-180)<=14:
                        for key in ret:
                            if key != 'corners':
                                name = hvstr+tday1+'M'+key
                                if name in out:
                                    out[name] = out[name]+ret[key]
                                else:
                                    out[name] = ret[key].copy()
                                # if np.isnan(out[name]).any():
                                #     print(f"{name} bads={np.isnan(out[name]).sum()} goods={np.isfinite(out[name]).sum()}")
                                # else:
                                if name[-2:] == '_c':
                                    print(f"{name[:-2]} {(out[name[:-2]]/out[name]).mean():.4}")
                            elif hvstr+key not in out:
                                out[hvstr+key] = ret[key]
    for key in out:
        if key+'_c' in out:
            out[key] = out[key]/out[key+'_c']
    for key in [k for k in out.keys() if k[-2:]=='_c']:
        out.pop(key)
    return out

def interpfields(arx, df, reqdates, reqdays, rads, verbose=False, keys=keys):
    x, y = sinu(df['longitude'].values, df['latitude'].values)
    for key in arx:
        if key[-7:] != 'corners':
            hvstr = '_'.join(key.split('_')[:3])+'_'
            x0,x1,y0,y1 = arx[hvstr+'corners']
            sz = arx[key].shape
            ok = (x<=x1) & (y>=y1) & (x>=x0) & (y<=y0) 
            xm = np.linspace(x0,x1,sz[0])
            ym = np.linspace(y0,y1,sz[1])
            xs=x[ok]; ys=y[ok]
            print(f"{key} count={ok.sum()}")
            ex = np.isfinite(arx[key]).astype(np.float32)
            fld = np.nan_to_num(arx[key],0.)
            key1 = key[len(hvstr):]
            for r in rads:
                v = getaver (xm,ym,xs,ys,fld,ex,r)
                name = key1[:10].replace('_','-')+key1[10:]+str(r)
                if name in df:
                    v0 = df[name][ok]
                    vi = np.isnan(v)
                    v[vi] = v0[vi]
                else:
                    df[name] = np.full_like(x, np.nan)
                df[name][ok] = v
    return df

if __name__ == '__main__':
    # import geojson
    # import pandas as pd
    # from os import path
    # workdir = 'evaluation'
    # # workdir = 'development'
    # with open(path.join(workdir,'grid_cells.geojson')) as f:
    #     grid = geojson.load(f)
    # grid = pd.DataFrame([{'cell_id':g['properties']['cell_id'], 'region':g['properties']['region'], 
    #                       'corners': np.array(g['geometry']['coordinates'])} for g in grid['features']]).set_index('cell_id')
    # gll = np.vstack(grid['corners'].values)    
    # grid['latitude'] = gll[:,:,1].mean(1)
    # grid['longitude'] = gll[:,:,0].mean(1)
    # stmeta = pd.read_csv(path.join(workdir,'ground_measures_metadata.csv')).set_index('station_id')
    
    import matplotlib.pyplot as plt
    file = getmodis_lat_lon('MYD10A1', 41.881832, -87.623177, datetime(2010, 5, 15))
    print(list(file.datasets().keys()))    
    rgb,x0,x1,y0,y1 = calibrate(file)
    for key in rgb:
        fig = plt.figure(frameon=False); ax = plt.Axes(fig,[0., 0., 1., 1.])
        ax.set_axis_off(); fig.add_axes(ax)
        plt.imshow(rgb[key])
        ax.set_title(key)