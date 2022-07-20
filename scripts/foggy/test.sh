#!/bin/bash

set -x
set -e

# 42.7(7)
CUDA_VISIBLE_DEVICES=0 python test_net.py --dataset cityscape --net vgg16 --cuda \
--model_dir models/cityscape/vgg16/TIA/da_faster_rcnn_cityscape_7.pth