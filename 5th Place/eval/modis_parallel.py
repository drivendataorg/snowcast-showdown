#!/usr/bin/env python3
# Download Modis data for a given dataframe

def main(path_to_df, run_date, df_part_num):
    '''Function to pull and save Modis data for a given dataframe.'''
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    logging.info(f'run_date: {run_date}, part_num: {df_part_num}')

    lookback = 15

    # Need filter for max date to be one day ahead
    max_date = (datetime.datetime.strptime(run_date, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    min_date = (datetime.datetime.strptime(run_date, '%Y-%m-%d') - datetime.timedelta(days=lookback + 1)).strftime(
        '%Y-%m-%d')

    service_account = 'snocast-earth-engine-sa@snocast-340802.iam.gserviceaccount.com '
    credentials = ee.ServiceAccountCredentials(service_account,
                                               '/content/drive/MyDrive/snocast/eval/snocast-340802-c9acf704c825.json')
    ee.Initialize(credentials)

    # Import the MODIS Terra Snow Cover Daily Global 500m collection.
    terra = ee.ImageCollection('MODIS/006/MOD10A1')
    # Import the MODIS Aqua Snow Cover Daily Global 500m collection.
    aqua = ee.ImageCollection('MODIS/006/MYD10A1')

    terra_snow_cover = terra.select('NDSI_Snow_Cover').filterDate(min_date, max_date)
    aqua_snow_cover = aqua.select('NDSI_Snow_Cover').filterDate(min_date, max_date)
    terra_info = terra_snow_cover.getInfo()['features']
    aqua_info = aqua_snow_cover.getInfo()['features']
    logging.info('Terra min date: {}'.format(terra_info[0]['properties']['system:index']))
    logging.info('Terra max date: {}'.format(terra_info[-1]['properties']['system:index']))
    logging.info('Aqua min date: {}'.format(aqua_info[0]['properties']['system:index']))
    logging.info('Aqua max date: {}'.format(aqua_info[-1]['properties']['system:index']))

    df = pd.read_parquet(path_to_df)
    modis_cols = ['location_id', 'latitude', 'longitude']
    unique_ids = df[modis_cols]
    logging.info(unique_ids.shape)

    output_cols = ['date',
                   'longitude',
                   'latitude',
                   'time',
                   'NDSI_Snow_Cover']

    terra_list = []
    aqua_list = []
    terra_ids = []
    aqua_ids = []

    for idx, row in df.iterrows():
        if idx % 250 == 0:
          logging.info(idx)

        # Define a region of interest with a buffer zone of 500 m
        poi = ee.Geometry.Point(row['longitude'], row['latitude'])
        roi = poi.buffer(500)

        terra_data = terra_snow_cover.getRegion(roi, scale=500).getInfo()[1:]
        terra_ids.extend([row['location_id']]*len(terra_data))
        terra_list.extend(terra_data)

        aqua_data = aqua_snow_cover.getRegion(roi, scale=500).getInfo()[1:]
        aqua_ids.extend([row['location_id']]*len(aqua_data))
        aqua_list.extend(aqua_data)

    logging.info(idx)
    logging.info('Saving output for {} ...'.format(run_date))

    terra_df = pd.DataFrame(terra_list, columns=output_cols)
    terra_df['location_id'] = terra_ids

    aqua_df = pd.DataFrame(aqua_list, columns=output_cols)
    aqua_df['location_id'] = aqua_ids

    terra_df.to_parquet(f'/content/drive/MyDrive/snocast/eval/data/modis/modis_parts/modis_terra_{run_date}_{df_part_num}.parquet')
    aqua_df.to_parquet(f'/content/drive/MyDrive/snocast/eval/data/modis/modis_parts/modis_aqua_{run_date}_{df_part_num}.parquet')



if __name__ == '__main__':
    # execute only if run as script
    # import libraries
    import argparse
    import ee
    import pandas as pd
    import datetime
    import logging
    import sys

    # Initiate the parser
    parser = argparse.ArgumentParser(description='Get Modis data for dataframe')

    # Add arguments
    parser.add_argument('path_to_df', type=str, default='',
                        help='Path to dataframe to pull Modis data for')
    parser.add_argument('-d', '--run_date', type=str, default='',
                        help='The date to pull the data from')
    parser.add_argument('-n', '--df_part_num', type=str, default='',
                        help='The part number of the dataframe')

    # Read arguments from the command line
    args = parser.parse_args()

    path_to_df = args.path_to_df
    run_date = args.run_date
    df_part_num = args.df_part_num

    main(path_to_df, run_date, df_part_num)