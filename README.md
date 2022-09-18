# Learning with a Ever-Changing Ontology (NeurIPS 2022) [www](https://linzhiqiu.github.io/papers/leco/)

LECO: Learning with a Ever-Changing Ontology

## Repository Overview

This repository contains all the image classification code and experiments that appear in our paper for reproducibility.

## Get Started
We provide an environment yml file for conda user at [environment.yml](environment.yml). Or else, you may install torch(==1.6.0) from official site.

## Pretraining (Saving model initialization)
We provide [pretrain.py](pretrain.py) to save the model initialization file to ensure reproducibility. You may refer to [pretrain.sh](pretrain.sh) for examples of how to save an checkpoint from random initialization (used in our paper).

## Training for 2 time periods (TPs)
For CIFAR-LECO and iNat-LECO with two TPs, please refer to [train.py](train.py).

## Training 4 time periods (TPs)
For iNat-LECO with four TPs, please refer to [train_for_more_tps.py](train_for_more_tps.py).

## Data Visualization
You may visualize the long-tailed distribution of Semi-iNat at [SemiInatStats.ipynb](SemiInatStats.ipynb).