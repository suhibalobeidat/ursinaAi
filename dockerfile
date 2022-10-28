FROM nvidia/cuda:11.8.0-base-ubuntu22.04


RUN apt-get update && apt-get install -y \
      git cmake ffmpeg pkg-config \
      qtbase5-dev libqt5opengl5-dev libqt5opengl5-dev \
      libtinyxml-dev \
      libgl1-mesa-dev \
    && cd /opt \
    && apt-get clean && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y libboost-python-dev

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-dev \
    && ln -s -f /usr/bin/python3.7 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*



RUN apt-get update && apt-get install -y build-essential fakeroot libpng-dev \
libjpeg-dev libtiff-dev zlib1g-dev libssl-dev libx11-dev \
libgl1-mesa-dev libxrandr-dev libxxf86dga-dev libxcursor-dev \
bison flex libfreetype6-dev libvorbis-dev libeigen3-dev \
libopenal-dev libode-dev libbullet-dev nvidia-cg-toolkit \
libgtk2.0-dev libassimp-dev libopenexr-dev 

RUN git clone https://github.com/panda3d/panda3d.git


WORKDIR /panda3d

RUN python3 makepanda/makepanda.py --everything --installer --no-opencv --no-x11 --threads=6

RUN dpkg -i panda3d1.11_1.11.0_amd64.deb

WORKDIR /


RUN apt-get -y install python3-pip
#RUN apt-get update && apt-get install -y python3.10-venv
#RUN python3 -m venv .venv
#ENV PATH="/.venv/bin:$PATH"

RUN pip install --upgrade \
    pip \
    setuptools \
    setproctitle \
    lz4 \
    psutil

RUN pip install --upgrade torch torchvision torchaudio
RUN pip install --upgrade pyperclip
RUN pip install --upgrade sklearn
RUN pip install --upgrade opencv-python
RUN pip install --upgrade h5py
RUN pip install --upgrade tensorboard
RUN pip install --upgrade einops
RUN pip install --upgrade python-xlib
RUN pip install psd-tools3==1.8.2

ENV PYTHONUNBUFFERED 1

RUN mkdir /navigation-ursina
WORKDIR /navigation-ursina


COPY . /navigation-ursina

#CMD [ "python3", "/navigation-ursina/train_ursina.py" ]