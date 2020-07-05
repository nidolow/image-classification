#!/bin/bash

echo $'WARNING: files created within this docker will be owned root. Remember to chown them.\n'
docker run --gpus all -it -v $PWD/:/tf/work -w /tf/work tensorflow/tensorflow:latest-gpu-py3 bash -c "pip install -r requirements.txt; $*"
