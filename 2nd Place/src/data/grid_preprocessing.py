import os
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


INPUT_FOLDER = 'C:/Users/wrkstation/projects/snowcast/data/input/'
OUTPUT_FOLDER = 'C:/Users/wrkstation/projects/snowcast/data/processed/'

def geoclustering(df, eps, earth_radius=6371.0088):
    epsilon = eps / earth_radius
    coords = list(zip(df.geometry.y.values, df.geometry.x.values))
    coords = np.deg2rad(coords)

    db = DBSCAN(eps=epsilon, min_samples=20, algorithm='ball_tree', metric='haversine')
    df['cluster'] = db.fit_predict(coords)

    return df


def process_grid():
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
    grid1 = gpd.read_file(os.path.join(INPUT_FOLDER, 'grid_cells.geojson'))
    grid2 = gpd.read_file(os.path.join(INPUT_FOLDER, 'grid_cell_stage2.geojson'))

    grid = pd.concat([grid1, grid2])
    grid = grid.drop_duplicates('cell_id')

    grid['geometry'] = grid['geometry'].centroid
    df = geoclustering(grid, eps=5)

    clusters = [[i, df[df['cluster'] == i].unary_union.convex_hull] for i in df['cluster'].unique() if i != -1]
    clusters = pd.DataFrame(clusters, columns=['cluster', 'geometry'])
    clusters = gpd.GeoDataFrame(clusters, geometry='geometry')
    clusters.set_crs(grid.crs)

    clusters.to_file(os.path.join(OUTPUT_FOLDER, 'grid_clusters.geojson'), driver='GeoJSON')
    df.to_file(os.path.join(OUTPUT_FOLDER, 'grid_cells_points.geojson'), driver='GeoJSON')

def main():
    process_grid()


if __name__ == '__main__':
    main()

