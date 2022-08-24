"""
DAG for loading data from HRRR and MODIS blob containers.
"""

import os
import logging
import requests
from glob import glob
from pathlib import Path
import pendulum

from datetime import datetime, date, timedelta
from azure.storage.blob import ContainerClient

from airflow import DAG
from airflow.decorators import task
from airflow import AirflowException
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'leigh',
    'start_date'            : "2021-09-01 10:00:00",
    'end_date'              : "2022-03-31 14:00:00",
    'retries': 30,
    'retry_delay': timedelta(hours=5),
}

data_dir = '/data'

blob_container = "https://noaahrrr.blob.core.windows.net/hrrr"
blob_container_aws = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
sector = "conus"
forecast_hour = 0   # offset from cycle time
product = "wrfsfcf" # 2D surface levels

variables12 = [
    'REFC:entire atmosphere',
    'UGRD:80 m above ground',
    'VGRD:80 m above ground',
    'TMP:surface:anl',
    'WEASD:surface:anl',
    'PWAT:entire atmosphere (considered as a single layer)'
]

variables00 = [
    'TMP:surface',
    'WEASD:surface:0-0',
]

def loader(url, filename, idx, variables, timeout=1.):
    r = b""
    try:
        for variable in variables:
            variable_idx = [l for l in idx if f":{variable}" in l][0].split(":")
            if len(variable_idx) == 0: continue
            line_num = int(variable_idx[0])
            range_start = variable_idx[1]
            next_line = idx[line_num].split(':') if line_num < len(idx) else None
            range_end = next_line[1] if next_line else None
            headers = {"Range": f"bytes={range_start}-{range_end}"}
            resp = requests.get(url, headers=headers, stream=True)
            if resp.ok:
                r += resp.content
        if r == b"":
            return False
        with open(filename, 'wb') as output:
            output.write(r)
        return True
    except:
        return False

with DAG(
        dag_id='loader_data',
        schedule_interval="0 18 * * *",
        catchup=True,
        tags=['downloads'],
        default_args=default_args,
    ) as dag:

    # Generate 6 modis tasks, for downloading h/v chunks
    for (h, v) in [(8,4),(8,5),(9,4),(9,5),(10,4),(10,5)]:
        @task(task_id=f'load_modis_h{h:0>2d}_v{v:0>2d}')
        def load_modis(h, v):
            day = datetime.fromisoformat(os.environ.get("AIRFLOW_CTX_EXECUTION_DATE"))
            day = day - timedelta(days=0)
            modis_cc = ContainerClient(
                account_url = 'https://modissa.blob.core.windows.net/',
                container_name = 'modis-006', credential=None)

            folder = "MYD10A1" + '/' + '{:0>2d}/{:0>2d}'.format(h,v) + '/' + f"{day:%Y%j}"
            filenames = glob(f"{data_dir}/modis/{folder}/**.hdf")
            if len(filenames) > 0: return True # file exist
            files = []
            generator = modis_cc.list_blobs(name_starts_with=folder)
            for blob in generator:
                if blob.name.endswith('.hdf'):
                    filename = os.path.join(data_dir, 'modis', blob.name)
                    print(filename)
                    if os.path.isfile(filename): return True
                    Path(filename).parent.mkdir(parents=True, exist_ok=True)
                    url = "https://modissa.blob.core.windows.net/modis-006/"+blob.name
                    resp = requests.get(url, stream=True)
                    if resp.ok:
                        with open(filename, 'wb') as output:
                            output.write(resp.content)
                    return True

            raise AirflowException

        modis_task = load_modis(h, v)

    @task(task_id=f'load_hrrr_time_12')
    def load_hrrr_t12():
        day = datetime.fromisoformat(os.environ.get("AIRFLOW_CTX_EXECUTION_DATE"))
        day = day - timedelta(days=0)
        for cycle in range(12, 9, -1):
            # Put it all together
            file_path = f"hrrr.t{cycle:02}z.{product}{forecast_hour:02}.grib2"
            parent = os.path.join(data_dir, 'hrrr', f"hrrr.{day:%Y%m%d}/{sector}")
            filename = os.path.join(parent, f"{file_path}")
            if os.path.isfile(filename): return True
            os.makedirs(parent, exist_ok=True)
            url = f"{blob_container}/hrrr.{day:%Y%m%d}/{sector}/{file_path}"
            print("URL BLOB ", url)
            r = requests.get(f"{url}.idx")
            if not r.ok:
                url = f"{blob_container_aws}/hrrr.{day:%Y%m%d}/{sector}/{file_path}"
                print("URL AWS ", url)
                r = requests.get(f"{url}.idx")
            idx = r.text.splitlines()
            if loader(url, filename, idx, variables12, timeout=1.): return True

        raise AirflowException

    hrrr_t12 = load_hrrr_t12()

    @task(task_id=f'load_hrrr_time_00')
    def load_hrrr_t00():
        day = datetime.fromisoformat(os.environ.get("AIRFLOW_CTX_EXECUTION_DATE"))
        day = day - timedelta(days=0)
        for cycle in range(0, 3):
            # Put it all together
            file_path = f"hrrr.t{cycle:02}z.{product}{forecast_hour:02}.grib2"
            parent = os.path.join(data_dir, 'hrrr', f"hrrr.{day:%Y%m%d}/{sector}")
            filename = os.path.join(parent, f"{file_path}")
            if os.path.isfile(filename): return True
            os.makedirs(parent, exist_ok=True)
            url = f"{blob_container}/hrrr.{day:%Y%m%d}/{sector}/{file_path}"
            print("URL BLOB ", url)
            r = requests.get(f"{url}.idx")
            if not r.ok:
                url = f"{blob_container_aws}/hrrr.{day:%Y%m%d}/{sector}/{file_path}"
                print("URL AWS ", url)
                r = requests.get(f"{url}.idx")
            idx = r.text.splitlines()
            print(idx)
            if loader(url, filename, idx, variables00, timeout=1.): return True
        print(url)
        raise AirflowException

    hrrr_t00 = load_hrrr_t00()
