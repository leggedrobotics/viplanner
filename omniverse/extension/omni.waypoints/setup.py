# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'omni.isaac.waypoints' python package."""


from setuptools import setup

# Installation operation
setup(
    name="omni-isaac-waypoints",
    author="Pascal Roth",
    author_email="rothpa@ethz.ch",
    version="0.0.1",
    description="Extension to extract waypoints in 3D environments.",
    keywords=["robotics"],
    include_package_data=True,
    python_requires="==3.7.*",
    packages=["omni.isaac.waypoints"],
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.7"],
    zip_safe=False,
)

# EOF
