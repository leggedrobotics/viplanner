# Training and Evaluation

Here an overview of the steps involved in training the policy is provided.


## Data Generation

For the data generation, please follow the instruction given in [here](omniverse/README.md).


## Cost-Map Building

Cost-Map building is an essential step in guiding optimization and representing the environment.
Cost-Maps can be built from either depth and semantic images (i.e., data generated in simulation) or (semantically annotated) point clouds (i.e., real-world data).

If depth and semantic images of the simulation are available, then first 3D reconstruction has to be performed, following the steps described in Point 1. If the (semantically annotated) pointclouds are generated, then the cost-map can be build directly from the pointcloud, following the steps described in Point 2.

1. **Simulation: Depth Reconstruction** <br>

    The reconstruction is executed in two steps, controlled by the config parameter defined in [ReconstructionCfg Class](viplanner/config/costmap_cfg.py):
    1. Generate colored point cloud by warping each semantic images onto the depth image (account for cameras in different frames)
    2. Projection into 3D space and voxelization

    The process expects following datastructure:

    ``` graphql
    env_name
    ├── camera_extrinsic.txt                    # format: x y z qx qy qz qw
    ├── intrinsics.txt                          # expects ROS CameraInfo format --> P-Matrix
    ├── depth                                   # either png and/ or npy, if both npy is used
    |   ├── xxxx.png                            # images saved with 4 digits, e.g. 0000.png
    |   ├── xxxx.npy                            # arrays saved with 4 digits, e.g. 0000.npy
    ├── semantics                               # optional
        ├── xxxx.png                            # images saved with 4 digits, e.g. 0000.png
    ```

    In the case that the semantic and depth images have an offset in their position (as typical on some robotic platforms),
    define a `sem_suffic` and `depth_suffix` in `ReconstructionCfg` to differentiate between the two with the following structure:

    ``` graphql
    env_name
    ├── camera_extrinsic{depth_suffix}.txt      # format: x y z qx qy qz qw
    ├── camera_extrinsic{sem_suffix}.txt        # format: x y z qx qy qz qw
    ├── intrinsics.txt                          # P-Matrix for intrinsics of depth and semantic images (depth first)
    ├── depth                                   # either png and/ or npy, if both npy is used
    |   ├── xxxx{depth_suffix}.png              # images saved with 4 digits, e.g. 0000.png
    |   ├── xxxx{depth_suffix}.npy              # arrays saved with 4 digits, e.g. 0000.npy
    ├── semantics                               # optional
        ├── xxxx{sem_suffix}.png                # images saved with 4 digits, e.g. 0000.png
    ```

2. **Real-World: Open3D-Slam**

    To create an annotated 3D Point-Cloud from real-world data, i.e., LiDAR scans and semantics generated from the RGB camera stream, use tools such as [Open3D Slam](https://github.com/leggedrobotics/open3d_slam).


3. **Cost-Building** <br>

    Either a geometric or semantic cost map can be generated running the following command:

    ```
    python viplanner/cost_builder.py
    ```

    With configs set in [CostMapConfig](viplanner/config/costmap_cfg.py). We provided some standard values, however, before running the script, please adjust the config to your needs and local environment paths.

    Cost-Maps will be saved within the environment folder, with the following structure:

    ``` graphql
    maps
    ├── cloud
    │   ├── cost_{map_name}.txt                 # 3d visualization of cost map
    ├── data
    │   ├── cost_{map_name}_map.txt             # cost map
    │   ├── cost_{map_name}_ground.txt          # ground height estimated from pointcloud
    └── params
        ├── config_cost_{map_name}.yaml         # CostMapConfig used to generate cost map

    ```


## Training

Configurations of the training given in [TrainCfg](viplanner/config/learning_cfg.py). Training can be started using the example training script [train.py](viplanner/train.py).

``` bash
python viplanner/train.py
```

For the training a directory structure as follows is expected/ will be created:

``` graphql
file_path                                       # TrainCfg.file_path or env variable EXPERIMENT_DIRECTORY
├── data
│   ├── env_name                                # structure as defined in Cost-Map Building
├── models
│   ├── model_name
│   |   ├── model.pth                           # trained model
│   |   ├── model.yaml                          # TrainCfg used to train model
├── logs
│   ├── model_name
```

It is important that the model name is unique, otherwise the previous training will be overwritten.
Also always copy the `model.pt` and `model.yaml` because the configs are necessary to reload the model.
