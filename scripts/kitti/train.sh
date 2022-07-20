#!/bin/bash

set -x

CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset kitti2cityscape --net vgg16 --cuda \
--epochs 5 --gamma 3.0 --context --preserve \
--alpha1 0.1 --alpha2 0.1 --alpha3 0.1 \
--lamda1 1.0 --lamda2 1.0 --lamda3 0.01 \
--num_aux1 2 --num_aux2 4 --desp 'TIA'