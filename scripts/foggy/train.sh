#!/bin/bash

set -x
set -e

CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset cityscape --net vgg16 --cuda \
--epochs 10 --gamma 3.0 --warmup --context --db --init \
--alpha1 0.1 --alpha2 1.0 --alpha3 1.0 \
--lamda1 1.0 --lamda2 1.0 --lamda3 0.01 \
--num_aux1 2 --num_aux2 4 --desp 'TIA'