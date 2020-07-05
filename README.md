# Introduction

Simple image classification project using Tensorflow. It allows to train DNN model
and classify images. Contains pretrained models for cat/dog/human recognition.


# Prerequistions

Project was developed with Linux and it is recommended as an OS.

Training was performed using Tensorflow 2.2.0 and for simplicity it is best
to use Docker. For docker installation refer to:
https://docs.docker.com/engine/install/. In case of problems here are some quite
useful post-installation steps:
https://docs.docker.com/engine/install/linux-postinstall/.

About Tensorflow docker images can be read here:
https://www.tensorflow.org/install/docker.
If you want to use GPU within docker follow this instruction:
https://github.com/NVIDIA/nvidia-docker.

Note: file requirements.txt is generated for tensorflow docker image! 

<div id="data"></div>


# Train data

Data set for pretrained models is not available, since it was delivered as part of
closed task from outside source. Data organization for training is as follows:

```
data
└───train
│   └───category1
│       │   file1.jpg
│       │   file2.jpg
│       │   ...
│   └───category2
│       │   file3.jpg
│       │   file4.jpg
│       │   ...
│   └───category3
│       │   file5.jpg
│       │   file6.jpg
│       │   ...   
``` 

where "category1", "category", "category3" are respectively names of data classes.

Original data consisted of:
* 12000 files of cats
* 11000 files of dogs
* 12232 files of human faces

Data was split and 80% was used for training, 10% for validation and 10% for test.


# Usage

To start working with tensorflow in docker run:

```
./scripts/run_in_docker.sh bash
```

or if you prefer to utilze GPU:

```
./scripts/run_in_docker_gpu.sh bash
```

## Training

For training run:

```
./scripts/run_in_docker.sh python ./src/train.py --arch vgg_v1
```

For more training configuration options refer to:

```
./scripts/run_in_docker.sh python ./src/train.py -h

``` 
Remember to have training [data](#data) set up.
Training results will be located in folder

```
models/
```

To get more detailed evaluation on test set separated from traing data, run:

```
./scripts/run_in_docker.sh python ./src/evaluate.py -m models/model-HASH.mdl
```

or to evaluate all models in folder:

```
./scripts/run_in_docker.sh python ./src/evaluate.py -d models/
```
## Predictions

To classify some images run:

```
./scripts/run_in_docker.sh python ./src/predict.py -m models/model-final.mdl -i image.jpg
```
