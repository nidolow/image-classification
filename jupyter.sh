#!/bin/bash

while getopts ":g" opt; do
  case ${opt} in
    g) docker run --gpus all -it -p 8888:8888 -u $(id -u):$(id -g) -v $PWD/:/tf/work -w /tf/work tensorflow/tensorflow:latest-gpu-py3-jupyter
      exit;;
    \?) echo "Usage: cmd [-g]"
      exit;;
  esac
done

docker run -it -p 8888:8888 -u $(id -u):$(id -g) -v $PWD/:/tf/work -w /tf/work tensorflow/tensorflow:latest-py3-jupyter
