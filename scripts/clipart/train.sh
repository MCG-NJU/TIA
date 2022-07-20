#!/bin/bash

set -x
set -e

CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset clipart --net res101 --cuda \
--epochs 12 --gamma 5.0 --warmup --context \
--alpha1 1.0 --alpha2 1.0 --alpha3 1.0 \
--lamda1 1.0 --lamda2 1.0 --lamda3 0.01 \
--num_aux1 2 --num_aux2 4 --desp 'TIA'