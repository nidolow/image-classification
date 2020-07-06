#!/bin/bash

echo $'WARNING: files created within this docker will be owned by root.\n'
docker run -it -v $PWD/:/tf/work -w /tf/work tensorflow/tensorflow:latest-py3 bash -c "pip install -r requirements.txt; $*"
