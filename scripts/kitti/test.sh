#!/bin/bash

set -x

# 43.7(5)
CUDA_VISIBLE_DEVICES=0 python test_net.py --dataset kitti2cityscape --net vgg16 --cuda \
--model_dir models/kitti2cityscape/vgg16/TIA/da_faster_rcnn_kitti2cityscape_7.pth