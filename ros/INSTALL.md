# INSTALL

It is strongly recommend to use the provided docker images for NVIDIA Jetson Orion (L4T r35.1.0) due to special version requirements on the Jetson!

## Models

For the models, place them in the `ros/planner/models` folder, both the viplanner and mask2former model.
For the semantics, we use the Mask2Former implementation of [mmdetection](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask2former), as it improves inference speed on the jetson compared to the code version publish by the authors. For inference reason, we use the smallest network with ResNet 50 backbone pre-trained on the COCO dataset that can be downloaded [here](https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_r50_lsj_8x2_50e_coco-panoptic/mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth).


## Nvidia Jetson Docker

Before building the docker images, enabling of Docker Default Runtime is necessary in otder to allow access to the CUDA compiler (nvcc) during `docker build` operations. Therefore, add `"default-runtime": "nvidia"` to your `/etc/docker/daemon.json` configuration file before attempting to build the containers:

```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },

    "default-runtime": "nvidia"
}
```

You will then want to restart the Docker service or reboot your system before proceeding. This can be done by running:
> service docker restart

In order to build the docker container on a NVIDIA Jetson Orin, execute the following steps:

1. On the jetson, local the files: `nvidia-l4t-apt-source.list` (usually at location `/etc/apt/sources.list.d/nvidia-l4t-apt-source.list`) and `nvidia-l4t-apt-source.clean.list` (if not on device, just creat an empty file). Then, copy both into `bin/packages`.

2. The DockerContext of the image is the parent directory of viplanner, thus, make sure you put the repo under a folder, e.g., git and not under your home as otherwise all files under home are copied to the context. The container can then be build as follows:
    ```bash
    ./bin/build.sh
    ```

3. To run the container, we assume that there exists `$HOME/catkin_ws` and `$HOME\git`. In the former, the catkin workspace with the `src` directory is located (don't build the packages yet) and in the second are any kind of git repositories that should be included. If both are there, the docker container can be started with:
    ```bash
    ./bin/run.sh
    ```

4. The viplanner repo should be linked into `$HOME/catkin_ws/src`. Then the planner's ROS Node can be build as follows:
    ```bash
    catkin build viplanner_pkgs
    ```
    Similarly add all other robot specific repositories and build the corresponidng packages.

**Remark**: If additional development environments such as TensorRT are required, you can add them in the docker file following the examples given in the [jetson-container repo](https://github.com/dusty-nv/jetson-containers).


## Manual Installation

- require ROS Noetic Installation (http://wiki.ros.org/noetic/Installation/Ubuntu)
- require CUDA Toolkit (same version as the one used to compile torch! This is crucial otherwise the segmentation network cannot run!)

- Dependency for JoyStick Planner:
  ```bash
  sudo apt install libusb-dev
  ```

- Installation of VIPlanner
    follow instructions in [README.md](../README.md) and install with inference flag. This installs mmdetection for Mask2Former, detailed instructions of the installation are given i the official documentation, [here](https://mmdetection.readthedocs.io/en/latest/).

- Build all ros packages
    ```bash
    catkin build viplanner_pkgs
    ```

## Known Issues

### ROS numpy

- In ROS numpy, there still exists `np.float` of previous numpy versions.
  - FIX:
    in '/opt/ros/noetic/lib/python3/dist-packages/ros_numpy/point_cloud2.py' change all occurrences of 'np.float' to 'float'

### General

- Setuptools version during install. VIPlanner requires are rather recent version of setuptools (>64.0.0) which can lead to problems with the mask2former install. It is recommended to always install mask2former first and then upgrade setuptools to the version needed for the VIPlanner. Otherwise following errors can be observed:
  - ERROR:
    ```bash
    Invalid version: '0.23ubuntu1'
    ```
  - FIX:
    > pip install --upgrade --user setuptools==58.3.0

  - ERROR:
    ```
    File "/usr/local/lib/python3.8/dist-packages/pkg_resources/_vendor/packaging/version.py", line 264, in __init__
        match = self._regex.search(version)
    TypeError: expected string or bytes-like object
    ```
  - FIX:
    manually editing `site-packages/pkg_resources/_vendor/packaging/version.py` with `str()`


### Within the Docker
- SSL Issue when running `pip install` within the docker when trying to manually install additional packages (description [here](https://stackoverflow.com/questions/50692816/pip-install-ssl-issue))

    - ERROR:
    ```bash
    python -m pip install torch
    Collecting zeep
        Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLError("bad handshake: Error([('SSL routines', 'ssl3_get_server_certificate', 'certificate verify failed')],)",),)': /simple/torch/
    ```

    - FIX:
    > python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --index-url=https://pypi.org/simple/ torch

- PyYAML upgrade error (described [here](https://clay-atlas.com/us/blog/2022/07/23/solved-cannot-uninstall-pyyaml-it-is-a-distutils-installed-project-and-thus-we-cannot-accurately-determine-which-files-belong-to-it-which-would-lead-to-only-a-partial-uninstall/))
    - ERROR:
    ```
    Cannot uninstall 'PyYAML'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.
    ```
    - FIX:
    > pip install --ignore-installed PyYAML

-
