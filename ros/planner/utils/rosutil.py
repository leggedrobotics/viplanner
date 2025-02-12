# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# python
import os

import numpy as np

# ROS
import rospy
import torch


class ROSArgparse:
    def __init__(self, relative=None):
        self.relative = relative

    def add_argument(self, name, default, type=None, help=None):
        name = os.path.join(self.relative, name)
        if rospy.has_param(name):
            rospy.loginfo("Get param %s", name)
        else:
            rospy.logwarn("Couldn't find param: %s, Using default: %s", name, default)
        value = rospy.get_param(name, default)
        variable = name[name.rfind("/") + 1 :].replace("-", "_")
        if isinstance(value, str):
            exec(f"self.{variable}='{value}'")
        else:
            exec(f"self.{variable}={value}")

    def parse_args(self):
        return self


def msg_to_torch(data, shape=np.array([-1])):
    return torch.from_numpy(data).view(shape.tolist())


def torch_to_msg(tensor):
    return [tensor.view(-1).cpu().numpy(), tensor.shape]


# EoF
