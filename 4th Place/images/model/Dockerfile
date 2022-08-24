FROM debian:bullseye-slim

RUN apt-get update \
    && apt-get install --no-install-recommends -yqq wget

# Install miniconda
ENV CONDA_DIR "/opt/conda/"
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget --quiet --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-py39_4.11.0-Linux-x86_64.sh \
   -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR} && rm ~/miniconda.sh

COPY ./environment.yaml /environment/
RUN conda env update --file /environment/environment.yaml && conda clean -afy
