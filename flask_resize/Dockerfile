FROM ubuntu:16.04

COPY . /usr/app/

EXPOSE 5000

WORKDIR /usr/app

RUN apt-get update -y && apt-get install -y \
    git \
    wget \
    curl \
    cmake \
    unzip \
    software-properties-common \
    unixodbc-dev \
    gcc \
    g++ \
    libsm6 \
    libxext6 \
    libxrender-dev 

# Install python

RUN apt-get update && apt-get -y upgrade
RUN apt-get install --no-install-recommends -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install --no-install-recommends -y python3.6
RUN apt-get install -y --no-install-recommends python3.6-dev gcc gfortran musl-dev

RUN apt-get -y install sudo
RUN sudo apt-get -y install python3-pip 
RUN sudo -H pip3 install --upgrade pip

RUN apt-get -y install libglib2.0-0 
RUN apt-get -y install libsm6 \ 
    libxrender-dev \ 
    libxext6 \
    ffmpeg

RUN sudo apt -y install python3-pandas

RUN pip3 install \
    gunicorn==19.9.0\
    Flask==1.1.2 \
    numpy==1.18.5 \
    opencv-contrib-python \
    opencv-python-headless \
    pickleshare==0.7.5 \
    Pillow==7.1.2 \
    scikit-image==0.14.1 \
    scikit-learn \
    scipy \
    torch==1.0.0 \
    torchvision==0.2.1 

CMD python3 main.py
