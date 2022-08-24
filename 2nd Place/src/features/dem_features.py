import rasterio as rio
import numpy as np
from rasterio.merge import merge
from rasterio.plot import show
import glob
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
# from shapely.geometry import Point
from shapely.geometry import box
from sklearn.neighbors import BallTree
from tqdm.auto import tqdm

INPUT_FOLDER = '/home/vervan/projects/snowcast/snowcast/data/'


def get_tile_bounds(files):
    bounds = []
    for file in files:
        tile = rio.open(file)
        file_name = file.split('/')[-1]
        bound = box(*tile.bounds)
        bounds.append([file_name, bound])
        tile.close()
    bounds = pd.DataFrame(bounds, columns=['tile', 'geometry'])
    bounds = gpd.GeoDataFrame(bounds, geometry='geometry')
    bounds = bounds.set_crs(epsg=4326)
    return bounds


def dem_buffer_features(dem, df, buffers=[200, 500]):
    dem = dem.to_dataframe().reset_index()
    coords = list(zip(df['geometry'].x, df['geometry'].y))
    coords = np.deg2rad(coords)

    bt = BallTree(np.deg2rad(dem[['x', 'y']].values), metric='haversine')
    dist, idx_nn1 = bt.query(coords, k=1)

    for r in buffers:
        nearest_points = bt.query_radius(coords, r=r / 1000 / 6371)
        for i, idx in tqdm(enumerate(nearest_points)):
            tmp = dem['band_data'].loc[idx]
            df.loc[i, f'alt_r{r}_mean'] = tmp.mean()
            df.loc[i, f'alt_r{r}_max'] = tmp.max()
            df.loc[i, f'alt_r{r}_min'] = tmp.min()
    df['alt'] = dem['band_data'].loc[idx_nn1.flatten()].reset_index(drop=True)

    return df


if __name__ == '__main__':

    files = glob.glob(os.path.join(INPUT_FOLDER + 'external/dem_tiles/', '*.tif'))
    clusters = gpd.read_file(os.path.join(INPUT_FOLDER + 'processed/', 'grid_clusters.geojson'))
    grid = gpd.read_file(os.path.join(INPUT_FOLDER + 'processed/', 'grid_cells_points.geojson'))
    bounds = get_tile_bounds(files)
    tiles = gpd.sjoin(bounds, clusters)

    points = grid[grid['cluster'].isin([-1])]

    points_for_tiles = {'ca': grid[grid['cluster'].isin([0])],
                        'co': grid[grid['cluster'].isin([1, 2, 4, 5, 6, 7])],
                        'wa': grid[grid['cluster'].isin([3])]
                        }
    features_dem = []
    for reg in tqdm(tiles['cluster'].unique()):
        input_path = os.path.join(INPUT_FOLDER, 'processed/dem_mosaic/')
        input_path = os.path.join(input_path, f'dem_{reg}.tif')
        dem = xr.open_dataset(input_path, engine='rasterio')
        tmp = dem_buffer_features(dem, grid[grid['cluster']==reg], buffers=[200])
        features_dem.append(tmp)
        del dem

    for tile in tqdm(points['tile'].unique()):
        input_path = os.path.join(INPUT_FOLDER, 'external/dem_tiles/')
        input_path = os.path.join(input_path, tile)
        dem = xr.open_dataset(input_path, engine='rasterio')
        dem = dem.to_dataframe().reset_index()

        df = points[points['tile'] == tile]
        df = df.reset_index(drop=True)
        df = df.drop(['index_right', 'tile'], axis=1)
        tmp = dem_buffer_features(dem, df, buffers=[200])
        features_dem.append(tmp)
        del dem

    features_dem = pd.concat(features_dem)
    features_dem.to_csv('/home/vervan/projects/snowcast/snowcast/data/processed/features.csv', index=False)
