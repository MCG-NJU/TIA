#!/bin/bash

set -x
set -e

# 46.3(10)
CUDA_VISIBLE_DEVICES=0 python test_net.py --dataset clipart --net res101 --cuda \
--model_dir models/clipart/res101/TIA/da_faster_rcnn_clipart_10.pth