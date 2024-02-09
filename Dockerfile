# Dockerfile based on the https://github.com/dusty-nv/jetson-containers


ARG BASE_IMAGE=nvcr.io/nvidia/l4t-base:r35.1.0
FROM ${BASE_IMAGE}

#
# setup environment
#
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            ca-certificates \
		  gnupg2 \
          apt-utils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

#
# configure nvidia apt repo
#
COPY viplanner/bin/packages/nvidia-l4t-apt-source.list /tmp/apt/nvidia-l4t-apt-source.list
COPY viplanner/bin/packages/nvidia-l4t-apt-source.clean.list /tmp/apt/nvidia-l4t-apt-source.clean.list

#
# install CUDA Toolkit
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  cuda-toolkit-* \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

#
# install cuDNN
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  libcudnn*-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

#
# Install build tools, build dependencies and python
#
RUN apt-get update && apt-get upgrade -y &&\
    apt-get install -y \
        python3-pip \
        build-essential \
        cmake \
        git \
        curl \
        nano \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libxine2-dev \
        libglew-dev \
        libtiff5-dev \
        zlib1g-dev \
        libjpeg-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libpostproc-dev \
        libswscale-dev \
        libeigen3-dev \
        libtbb-dev \
        libgtk2.0-dev \
        pkg-config \
        lsb-release \
        gnupg2 \
        ca-certificates \
        figlet \
        ## Python
        python-dev \
        python-numpy \
        python3-dev \
        python3-pip \
        python3-numpy \
        python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --verbose numpy

#
# Install OpenCV
#
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopencv-dev=4.2.0+dfsg-5 \
        python3-opencv \
    && rm -rf /var/lib/apt/lists/*

#
# ROS Noetic
#
ENV ROS_DISTRO=noetic
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
ENV ROS_PYTHON_VERSION=3

WORKDIR /workspace

# add the ROS deb repo to the apt sources list
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# install ROS packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ros-noetic-desktop-full \
        ros-noetic-image-transport \
        ros-noetic-vision-msgs \
          python3-rosdep \
          python3-rosinstall \
          python3-rosinstall-generator \
          python3-wstool \
    && rm -rf /var/lib/apt/lists/*

# init/update rosdep
RUN apt-get update && \
    cd ${ROS_ROOT} && \
    rosdep init && \
    rosdep update && \
    rm -rf /var/lib/apt/lists/*

# Install catkin tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

# setup entrypoint
COPY viplanner/bin/packages/ros_entrypoint.sh /ros_entrypoint.sh
RUN echo 'source /opt/ros/${ROS_DISTRO}/setup.bash' >> /root/.bashrc
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
WORKDIR /

WORKDIR /root


#
# install prerequisites (many of these are for numpy)
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            python3-pip \
            python3-dev \
            libopenblas-dev \
            libopenmpi-dev \
            openmpi-bin \
            openmpi-common \
            gfortran \
            libomp-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip3 install --no-cache-dir setuptools Cython wheel
RUN pip3 install --no-cache-dir --verbose numpy

ENV LD_LIBRARY_PATH=/usr/lib/llvm-8/lib:$LD_LIBRARY_PATH

#
# PyTorch (for JetPack 5.0.1 DP)
#

# latest pytorch version that has distributed support for jetson (necessary for mmdet)
ARG PYTORCH_URL=https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl
ARG PYTORCH_WHL=torch-1.11.0-cp38-cp38-linux_aarch64.whl

RUN wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${PYTORCH_URL} -O ${PYTORCH_WHL} && \
    pip3 install --no-cache-dir --verbose ${PYTORCH_WHL} && \
    rm ${PYTORCH_WHL}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  software-properties-common \
		  apt-transport-https \
		  ca-certificates \
		  gnupg \
		  lsb-release \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# note:  cmake is currently pinned to 3.22.3 because of https://github.com/pytorch/pytorch/issues/74955
RUN pip3 install --upgrade --no-cache-dir --verbose cmake==3.22.3
RUN cmake --version

# patch for https://github.com/pytorch/pytorch/issues/45323
RUN PYTHON_ROOT=`pip3 show torch | grep Location: | cut -d' ' -f2` && \
    TORCH_CMAKE_CONFIG=$PYTHON_ROOT/torch/share/cmake/Torch/TorchConfig.cmake && \
    echo "patching _GLIBCXX_USE_CXX11_ABI in ${TORCH_CMAKE_CONFIG}" && \
    sed -i 's/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=")/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")/g' ${TORCH_CMAKE_CONFIG}

ENV LLVM_CONFIG="/usr/bin/llvm-config-9"

ARG MAKEFLAGS=-j$(nproc)
ARG PYTHON3_VERSION=3.8

RUN printenv

#
# install further requirements
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		liblapack-dev \
		libblas-dev \
		libhdf5-serial-dev \
		hdf5-tools \
		libhdf5-dev \
		zip \
		libjpeg8-dev \
		libopenmpi3 \
		protobuf-compiler \
		libprotoc-dev \
		llvm-9 \
		llvm-9-dev \
		libffi-dev \
		libsndfile1 \
        vim \
        libusb-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean


#
# python pip packages
#
RUN pip3 install --no-cache-dir --ignore-installed pybind11
RUN pip3 install --no-cache-dir --verbose ipython
RUN pip3 install --no-cache-dir --verbose opencv-python==4.5.5.64

#
# VIPlanner specific files
#

ENV CUDA_HOME='/usr/local/cuda'
RUN pip3 install --no-cache-dir --verbose mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html

# install viplanner in edible mode (needed update of pip and setuptools) --> update viplanner without rebuilding the image
COPY viplanner /viplanner
RUN pip3 install --upgrade pip
RUN pip3 install setuptools==66.0.0
RUN pip3 install git+https://github.com/cocodataset/panopticapi.git
# FIX for PyYAML 6.0.0 install error (see README.md)
RUN pip3 install --ignore-installed PyYAML==6.0.0
# FIX since pypose requires torch that is not available on jetson with cuda and torch.distributed available
# FIX pypose requires version 0.3.6 for torch < 2.0.x and work with toch 1.11
RUN pip3 install --no-dependencies pypose==0.3.6
ENV PATH="$PATH:/root/.local/bin"
RUN pip3 install --user -e /viplanner/.[inference,jetson]
RUN pip3 install pillow==9.4.0


# ros-numpy
RUN apt-get update
RUN apt-get install ros-noetic-ros-numpy
