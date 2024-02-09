"""Installation script for the 'omni.viplanner' python package."""


from setuptools import setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "numpy",
    "scipy>=1.7.1",
    # RL
    "torch>=1.9.0",
]

# Installation operation
setup(
    name="omni-isaac-viplanner",
    author="Pascal Roth",
    author_email="rothpa@ethz.ch",
    version="0.0.1",
    description="Extension to include ViPlanner: Visual Semantic Imperative Learning for Local Navigation",
    keywords=["robotics", "rl"],
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=INSTALL_REQUIRES,
    packages=["omni.viplanner"],
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.7"],
    zip_safe=False,
)

# EOF
