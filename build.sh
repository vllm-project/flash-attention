#!/bin/bash
# A simple build script for local testing.
# NOTE: This script is not used for the actual build process.

PYTORCH_VERSION="2.4.0"

pip install packaging ninja;
pip install torch==${PYTORCH_VERSION};
time python setup.py bdist_wheel --dist-dir=dist;
