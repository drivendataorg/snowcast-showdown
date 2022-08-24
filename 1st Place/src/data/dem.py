import cv2
import gc
from geotiff import GeoTiff
import numpy as np
from os import path,makedirs
import matplotlib.pyplot as plt
import planetary_computer
from pystac_client import Client
from scipy.io import savemat
import wget

def getdem (lat1=31.,lat2=51.,lon1=-126.,lon2=-102.,dir = 'dem', matfile='dem.mat'):
    makedirs(dir,exist_ok=True)
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                          ignore_conformance=True)
    cosd = lambda x: np.cos(np.radians(x))
    
    sz = 1200
    dsm = []
    for lon in np.arange(lon1+0.5,lon2):
        col = []
        for lat in np.arange(lat1+0.5,lat2):
            file = path.join(dir, f'Copernicus_DSM_COG_30_N{int(lat)}_00_W{int(1-lon)}_00_DEM.tif')            
            if not path.isfile(file):
                search = client.search(collections=["cop-dem-glo-90"],
                                        intersects={"type": "Point", "coordinates": [lon, lat]})
                items = list(search.get_items())        
                if len(items)>0:            
                    print(f"Returned {len(items)} items")
                    signed_asset = planetary_computer.sign(items[0].assets["data"])
                    fname = wget.download(signed_asset.href, dir)
                else:
                    print("Not found "+file)
            if path.isfile(file):
                file = GeoTiff(file)
                arr = np.array(file.read())[-1::-1]
                print([file.get_coords(0,0),file.get_coords(arr.shape[0]-1,arr.shape[1]-1)])
                if (arr.shape[1] != sz) or (arr.shape[0] != sz):
                   arr =  cv2.resize(arr,(sz,sz))
                # plt.imshow(arr)
            else:
                arr = np.zeros((sz,sz))
            col.append(arr)
        dsm.append (np.concatenate(col,axis=0))
    dsm = np.concatenate(dsm,axis=1)
    del col,arr
    gc.collect()
    
    rm = 2.
    r = 5
    sh = (r-1)//2
    av = dsm.reshape(dsm.shape[0]//r,r,-1,r).mean(-1).mean(1)
    x = dsm[1:]-dsm[:-1]
    south = x.copy()
    cv2.GaussianBlur(x[:,1:]+x[:,:-1], (2*int(rm*r)+1, 2*int(rm*r)+1), r, south, r)
    south = south[sh::r,sh::r]
    
    x = np.abs(x)
    x = 0.5*(x[:,1:]+x[:,:-1])
    y = dsm[:,1:]-dsm[:,:-1]
    del dsm
    gc.collect()
    y /= cosd(np.arange(lat1,lat2,1/sz))[:,None]
    east = y.copy()
    cv2.GaussianBlur(y[1:]+y[:-1], (2*int(rm*r)+1, 2*int(rm*r)+1), r, east, r)
    east = east[sh::r,sh::r]
    y = np.abs(y)
    y = 0.5*(y[1:]+y[:-1])    
    aspect = np.hypot(x,y)
    del x,y
    gc.collect()
    
    avs = aspect.copy()
    cv2.GaussianBlur(aspect, (2*int(rm*r)+1, 2*int(rm*r)+1), r, avs, r)
    aspect = avs[sh::r,sh::r]
    del avs
    
    lat = np.arange(lat1+(sh+0.5)/sz,lat2-sh/sz,r/sz)
    lon = np.arange(lon1+(sh+0.5)/sz,lon2-sh/sz,r/sz)
    
    out = {'lat': lat, 'lon': lon, 'aspect': aspect, 'elev': av, 'south': south, 'east': east}
    print (f'Saving DEM to {matfile}')
    savemat(matfile, out, do_compression=True)
    return out

if __name__ == '__main__':
    getdem()
    