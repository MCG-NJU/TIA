# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import, division, print_function

import numpy as np
from datasets.pascal_voc import pascal_voc
from datasets.imagenet import imagenet
from datasets.cityscape import cityscape
from datasets.cityscape_car import cityscape_car
from datasets.clipart import clipart
from datasets.kitti_car import kitti_car



__sets = {}


# Set up voc_<year>_<split>
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'trainval_aug', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

for year in ["2007"]:
    for split in ["trainval", "train", "test"]:
        name = "clipart_{}".format(split)
        __sets[name] = lambda split=split: clipart(split, year)

for year in ['2012']:
    for split in ["train", "train_aug"]:
        name = "kitti_car_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: kitti_car(split, year)

for year in ["2007"]:
    for split in ["train_s", "train_aug", "train_t", "test"]:
        name = "cityscape_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: cityscape(split, year)

for year in ["2007"]:
    for split in ["train_s", "train_t", "test_s", "test_t", "train_aug"]:
        name = "cityscape_car_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: cityscape_car(split, year)

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError("Unknown dataset: {}".format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())