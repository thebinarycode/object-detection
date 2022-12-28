FROM ubuntu:20.04

USER root

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y upgrade && apt-get autoremove

RUN apt-get install -y --no-install-recommends \
        build-essential \
        nvidia-cuda-toolkit \
        xdg-utils \
        apt-utils \
        cpio \
        curl \
        vim \
        git \
        lsb-release \
        pciutils \
        python3.8 \
        python3-pip \
        libgflags-dev \
        libboost-dev \
        libboost-log-dev \
        cmake \
        libx11-dev \
        libssl-dev \
        locales \
        libjpeg8-dev \
        libopenblas-dev \
        gnupg2 \
        protobuf-compiler \
        python3-dev \
        wget \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libusb-1.0-0-dev \
        sudo 

RUN pip3 install wheel
RUN pip3 install --upgrade pip

RUN mkdir -p /app/
RUN mkdir -p /app/object_detection/
RUN mkdir -p /app/object_detection/saved-model/

WORKDIR /app/object_detection/

COPY TrainModel.py /app/object_detection/
COPY requirements.txt /app/object_detection/
COPY dataset /app/object_detection/dataset/

#RUN pip install tensorflow tflite_model_maker==0.4.2 tflite_support pycocotools
RUN pip install -r requirements.txt

RUN python3 -V

RUN git clone https://github.com/tensorflow/tensorflow.git

CMD ["python3", "TrainModel.py", "efficientdet_lite0", "1", "1", "saved-model"]
CMD ["python3" "-m" "tensorflow.lite.tools.visualize" "./saved-model/android.tflite" "./saved-model/visualized_model.html"]
