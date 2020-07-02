#!/bin/bash

docker run -it -p 8888:8888 -u $(id -u):$(id -g) -v $PWD/:/tf/work -w /tf/work tensorflow/tensorflow:latest-py3-jupyter
