#!/bin/bash

# Determine the directory where the script resides
script_dir=$(dirname "${BASH_SOURCE[0]}")
absolute_dir=$(realpath "$script_dir")

# Set PYTHONPATH to include the script's directory (or any specific subdirectory)
export ACT_PROJECT_DIR=$absolute_dir
export PYTHONPATH=$PYTHONPATH:$absolute_dir
echo "ACT_PROJECT_DIR set to: $PYTHONPATH"
