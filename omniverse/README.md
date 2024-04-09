# ViPlanner Omniverse Extension

The ViPlanner Omniverse Extension offers a sophisticated testing environment for ViPlanner.
Within NVIDIA Isaac Sim as a photorealistic simulator, this extension provides an assessment tool for ViPlanner's performance across diverse environments.
The extension is developed using the [Orbit Framework](https://isaac-orbit.github.io/).

**Remark**

The extension for `Matterport` and `Unreal Engine` meshes with semantic information is currently getting updated to the latest Orbit version and will be available soon. An intermediate solution is given [here](https://github.com/pascal-roth/orbit_envs).

## Installation

To install the ViPlanner extension for Isaac Sim version 2023.1.1, follow these steps:

1. Install Isaac Sim using the [Orbit installation guide](https://isaac-orbit.github.io/orbit/source/setup/installation.html).
2. Clone the orbit repo, checkout commit `477cd6b3f` and link the viplanner extension. The specific commit is necessary as Orbit is under active development and the extension is not yet compatible with the latest version.

```
git clone git@github.com:NVIDIA-Omniverse/orbit.git
cd orbit
git checkout 477cd6b3f
cd source/extensions
ln -s {VIPLANNER_DIR}/omniverse/extension/omni.viplanner .
```

3. TEMPORARY: To use Matterport with semantic information within Isaac Sim, a new extension has been developed as part of this work. Currently, all parts are getting updated to the latest Orbit version. A temporary solution that is sufficient for the demo script is available [here](https://github.com/pascal-roth/orbit_envs). Please also clone and link it into orbit.

```
git clone git@github.com:pascal-roth/orbit_envs.git
cd orbit/source/extension
ln -s {ORBIT_ENVS}/extensions/omni.isaac.matterport .
```

4. Then run the orbit installer script and additionally install ViPlanner in the Isaac Sim virtual environment.

```
./orbit.sh -i -e
./orbit.sh -p -m pip install -e {VIPLANNER_DIR}
```

**Remark**
Also in orbit, it is necessary to comply with PEP660 for the install. This requires the following versions (as described [here](https://stackoverflow.com/questions/69711606/how-to-install-a-package-using-pip-in-editable-mode-with-pyproject-toml) in detail)
- [pip >= 21.3](https://pip.pypa.io/en/stable/news/#v21-3)
	```
  ./orbit.sh -p -m pip install --upgrade pip
  ```
- [setuptools >= 64.0.0](https://github.com/pypa/setuptools/blob/main/CHANGES.rst#v6400)
	```
  ./orbit.sh -p -m pip install --upgrade setuptools
  ```

## Usage

A demo script is provided to run the planner in three different environments: [Matterport](https://niessner.github.io/Matterport/), [Carla](https://carla.org//), and [NVIDIA Warehouse](https://docs.omniverse.nvidia.com/isaacsim/latest/features/environment_setup/assets/usd_assets_environments.html#warehouse).
In each scenario, the goal is represented as a movable cube within the environment.

To run the demo, download the model: [[checkpoint](https://drive.google.com/file/d/1PY7XBkyIGESjdh1cMSiJgwwaIT0WaxIc/view?usp=sharing)] [[config](https://drive.google.com/file/d/1r1yhNQAJnjpn9-xpAQWGaQedwma5zokr/view?usp=sharing)] and the environment files. Then adjust the paths (marked as `${USER_PATH_TO_USD}`) in the corresponding config files.

### Matterport
[Config](./extension/omni.viplanner/omni/viplanner/config/matterport_cfg.py)

To download Matterport datasets, please refer to the [Matterport3D](https://niessner.github.io/Matterport/) website. The dataset should be converted to USD format using Isaac Sim by executing the following steps:
1. Import the `.obj` file (located under `matterport_mesh`) into Isaac Sim by going to `File -> Import`.
2. Fix potential import setting such as Rotation and Scale. (`Property Panel -> Transform -> Rotate:unitsResolve = 0.0; Scale:unitsResolve = [1.0, 1.0, 1.0]`)
3. Export the scene as USD (`File -> Save as`).

```
./orbit.sh -p {VIPLANNER_DIR}/omniverse/standalone/viplanner_demo.py --scene matterport --model_dir {MODEL_DIR}
```

### Carla
[Download USD Link](https://drive.google.com/file/d/16OHwmEtSKBf36mh8VUpRFeHSqhbHzQgP/view?usp=sharing)  [Download Texture Link](https://drive.google.com/file/d/1jsvkObiLOwg_zoVTC4vO7JprETSHdb7N/view?usp=sharing) [Config](./extension/omni.viplanner/omni/viplanner/config/carla_cfg.py)

:warning: Due to some code changes, the semantics are here not correctly received. We are working on a fix.

```
./orbit.sh -p {VIPLANNER_DIR}/omniverse/standalone/viplanner_demo.py --scene carla --model_dir {MODEL_DIR}
```

### NVIDIA Warehouse
[Download USD Link](https://drive.google.com/file/d/1QXxuak-1ZmgKkxhE0EGfDydApVr6LrsF/view?usp=sharing) [Config](./extension/omni.viplanner/omni/viplanner/config/warehouse_cfg.py)
```
./orbit.sh -p {VIPLANNER_DIR}/omniverse/standalone/viplanner_demo.py --scene warehouse --model_dir {MODEL_DIR}
```

## Data Collection and Evaluation

Script for data collection and evaluation are getting updated to the latest Orbit version and will be available soon. If you are interested in the current state, please contact us.
