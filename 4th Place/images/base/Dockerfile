FROM apache/airflow:2.2.3-python3.8

USER root
RUN touch /var/run/docker.sock
RUN chown -R airflow /var/run/docker.sock
USER airflow 
