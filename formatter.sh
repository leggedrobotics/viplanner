#!/bin/bash

# get source directory
export VIPLANNER_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"


# run the formatter over the repository
# check if pre-commit is installed
if ! command -v pre-commit &>/dev/null; then
    echo "[INFO] Installing pre-commit..."
    pip install pre-commit
fi
# always execute inside the IsaacLab directory
echo "[INFO] Formatting the repository..."
cd ${VIPLANNER_PATH}
pre-commit run --all-files
cd - > /dev/null
