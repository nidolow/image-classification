#!/bin/bash

docker run -it -u $(id -u):$(id -u) -v $PWD/:/tf/work -w /tf/work tensorflow/tensorflow:latest-py3 bash -c "pip install -r requirements.txt; $*"
