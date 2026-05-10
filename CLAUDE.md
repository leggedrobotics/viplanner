# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 项目概述

ViPlanner 是一个基于学习的局部路径规划器，使用语义图像和深度图像在室内外环境中导航。完全在仿真中训练，通过 ROS 部署到真实腿足机器人（ANYmal）上。

## 安装

```bash
# 标准开发安装
pip install -e .[standard]

# 含推理支持（用于 ROS 部署）
pip install -e .[standard,inference]

# mmcv 通常需要指定 CUDA 版本安装（例如 CUDA 11.7 + torch 2.0）：
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html

# 可选：mask2former RGB 骨干网络（发布模型不需要）
pip install git+https://github.com/facebookresearch/detectron2.git
git submodule update --init
pip install -r third_party/mask2former/requirements.txt
cd third_party/mask2former/mask2former/modeling/pixel_decoder/ops && sh make.sh
```

CUDA 工具包未自动检测时需设置：
```bash
export CUDA_HOME=/usr/local/cuda
```

训练数据目录由 `TrainCfg.file_path` 控制，也可通过环境变量覆盖：
```bash
export EXPERIMENT_DIRECTORY=/path/to/data
```

## 代码检查与格式化

```bash
# 运行所有 pre-commit 钩子（black、flake8、isort、pyupgrade、codespell、许可证头）
./formatter.sh

# 或直接运行
pre-commit run --all-files
```

配置说明：black 行长 120 字符，flake8 忽略 E402/E501/W503/E203。`__init__.py` 文件免于 F401 检查。每个 `.py` 和 `.yml` 文件必须包含 BSD-3 许可证头（由 pre-commit 强制执行），模板位于 `.github/LICENSE_HEADER.txt`。

## 架构

### 核心 Python 包（`viplanner/`）

**网络（`plannernet/`）** — 采用双编码器 + 解码器架构：
- `PlannerNet` — 类 ResNet-18 骨干网络（4 阶段 BasicBlock），同时用作深度和语义编码器
- `DualAutoEncoder` — 融合深度与语义编码（拼接后 1024 通道），解码输出 `k` 个路径点（默认 5 个）和一个 fear 标量
- `RGBEncoder` — 可选的基于 Mask2Former 的编码器，用 RGB 输入替代语义图像
- 输出：`(waypoints [B, k, 3], fear [B, 1])`，坐标系为相机坐标系

**代价地图（`cost_maps/`）** — 以点云形式存储的可微 2D 代价地图：
- `SemCostMap` — 从语义标注点云构建
- `TsdfCostMap` — 从纯几何点云构建
- `CostMapPCD` — 训练时使用的统一内存表示；提供 `Pos2Ind` 支持可微网格采样

**轨迹优化（`traj_cost_opt/`）** — 命令式学习损失：
- `TrajCost.CostofTraj()` — 计算训练信号的组合损失（障碍物 + 高度 + 运动 + 目标），无需真值轨迹
- `TrajOpt` — 生成用于运动损失项的参考直线轨迹
- 使用 `pypose` SE3 操作将路径点变换到世界坐标系，再查询代价地图

**配置（`config/`）** — 基于 dataclass 的配置，支持 YAML 序列化/反序列化：
- `TrainCfg` — 所有训练超参数、路径、网络选项
- `DataCfg` — 数据加载、数据增强、训练/验证集划分
- `CostMapConfig` / `ReconstructionCfg` / `SemCostMapConfig` / `TsdfCostMapConfig` — 代价地图构建配置

**工具（`utils/`）** — `Trainer` 编排完整训练循环（数据集构建、优化器、WandB 日志、模型存档）。数据集样本为 `(depth_image, semantic_image, odometry, goal)` 四元组。

### 入口脚本

| 脚本 | 功能 |
|---|---|
| `viplanner/depth_reconstruct.py` | 从深度图和语义图生成彩色点云 |
| `viplanner/cost_builder.py` | 从点云构建语义或几何代价地图 |
| `viplanner/train.py` | 训练规划器（在脚本中配置 `TrainCfg`） |
| `viplanner/utils/eval_utils.py` | 评估指标 |

### 训练流程（顺序执行）

1. **数据采集** — 通过 IsaacLab（Matterport / Carla / Warehouse 环境）
2. **三维重建** — `depth_reconstruct.py`，使用 `ReconstructionCfg`；需要 `depth/`、`semantics/`、`camera_extrinsic.txt`、`intrinsics.txt`
3. **代价地图构建** — `cost_builder.py`，使用 `CostMapConfig`；输出至 `maps/cloud/`、`maps/data/`、`maps/params/`
4. **训练** — `train.py`；输出至 `{EXPERIMENT_DIRECTORY}/models/{model_name}/`；模型名称必须唯一，且须同时保存 `model.pth` 和 `model.yaml`
5. **评估** — 在 ROS 或 Isaac Sim 中使用保存的模型文件对

### 部署

**ROS（`ros/`）** — 面向 ANYmal 的 ROS Noetic 包：
- `planner/` — 运行语义分割 + ViPlanner 推理
- `pathFollower/` — 将路径点转换为 twist 指令
- `visualizer/` — 在相机画面上叠加路径显示
- `waypoint_rviz_plugin/` — RViz 路径点设置工具
- 启动命令：`roslaunch viplanner_node viplanner.launch`

**Isaac Sim（`omniverse/`）** — IsaacLab 扩展（测试于 IsaacSim 4.2.0、IsaacLab 1.2.0）：
- 扩展位于 `omniverse/extension/omni.viplanner` 和 `omni.isaac.matterport`
- 需要软链接到 `IsaacLab/source/extensions/` 并通过 `./isaaclab.sh` 安装

### 关键设计决策

- **命令式学习**：无需真值轨迹标签，代价地图本身通过可微网格采样（`F.grid_sample`，`bicubic` 模式）提供训练信号
- **Fear 输出**：用 BCE 损失训练的二值碰撞概率头，推理时用于触发重规划
- **机器人足迹**：障碍物损失使用路径法向量将每个路径点横向扩展 `robot_width/2`，评估三条平行轨迹
- **语义 vs. 几何代价**：语义地图为每类别分配代价；几何（TSDF）地图仅使用几何信息；由 `CostMapConfig` 中的 `semantics` 标志选择
