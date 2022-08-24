import os
import requests
import geopandas as gpd
from pystac_client import Client
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from joblib import Parallel, delayed

INPUT_FOLDER = 'C:/Users/wrkstation/projects/snowcast/data/processed/'
OUTPUT_FOLDER = 'C:/Users/wrkstation/projects/snowcast/data/external/dem_tiles'



def get_dem_urls_poly(df):
    url_dem = "https://planetarycomputer.microsoft.com/api/stac/v1"
    client = Client.open(url_dem, ignore_conformance=True)

    urls = []
    for idx, poly in df.iterrows():
        coords = [[x, y] for x, y in poly['geometry'].exterior.coords]
        search = client.search(collections=["cop-dem-glo-30"], intersects={"type": "Polygon", "coordinates": [coords]})
        resp = search.get_all_items_as_dict()
        for i in resp['features']:
            urls.append(i['assets']['data']['href'])

    return urls

def get_dem_urls_point(point):
    url_dem = "https://planetarycomputer.microsoft.com/api/stac/v1"
    client = Client.open(url_dem, ignore_conformance=True)
    search = client.search(collections=["cop-dem-glo-30"], intersects={"type": "Point", "coordinates": point})
    resp = search.get_all_items_as_dict()
    url = resp['features'][0]['assets']['data']['href']

    return url

def download_tile(url, output_folder):
    session = requests.Session()
    retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))

    file_name = url.split('/')[-1]

    with open(os.path.join(output_folder, file_name), 'wb') as f:
        f.write(session.get(url).content)

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    grid = gpd.read_file(os.path.join(INPUT_FOLDER, 'grid_cells_points.geojson'))
    clusters = gpd.read_file(os.path.join(INPUT_FOLDER, 'grid_clusters.geojson'))
    points = grid[grid['cluster'] == -1]
    coords = list(zip(points.geometry.x.values, points.geometry.y.values))

    urls_poly = get_dem_urls_poly(clusters)
    urls_point = Parallel(n_jobs=-1)(delayed(get_dem_urls_point)(i) for i in tqdm(coords))

    urls = [*urls_poly, *urls_point]
    urls = list(set(urls))

    Parallel(n_jobs=-1)(delayed(download_tile)(i, OUTPUT_FOLDER) for i in tqdm(urls, desc='Download DEM tiles'))


if __name__ == '__main__':
    main()

