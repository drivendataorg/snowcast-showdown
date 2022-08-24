from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash_operator import BashOperator
from airflow.operators.docker_operator import DockerOperator
from docker.types import Mount
import requests
import re
import os
import time
import tempfile
import argparse
from glob import glob
from pathlib import Path
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

default_args = {
    'owner'                 : 'leigh',
    'description'           : 'Estimate SWE values',
    'start_date'            : "2022-02-03 00:00:00",
    'end_date'              : "2022-07-05 12:00:00",
    'depends_on_past'       : True,
    'email_on_failure'      : False,
    'email_on_retry'        : False,
    'retries'               : 12,
    'retry_delay'           : timedelta(hours=1)
}

with DAG( dag_id='predictor', schedule_interval="0 9 * * 5",
        catchup=True, default_args=default_args) as dag:

    day = os.environ.get("AIRFLOW_CTX_EXECUTION_DATE")
    estimator = DockerOperator(
        task_id='estimator',
        image='model:base',
        api_version='auto',
        auto_remove=True,
        mount_tmp_dir=False,
        tmp_dir='/tmp',
        command=["python", "/data/estimate.py", "-d", "{{ data_interval_end }}"],
        mounts=[
             Mount(
               source="/home/ubuntu/airflow/data",
               target="/data",
               type='bind',
              )
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge"
    )
