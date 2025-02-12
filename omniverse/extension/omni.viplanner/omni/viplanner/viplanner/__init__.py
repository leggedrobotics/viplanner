# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# from .vip_anymal import VIPlanner
import os

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))

from .viplanner_algo import VIPlannerAlgo

__all__ = ["DATA_DIR", "VIPlannerAlgo"]
