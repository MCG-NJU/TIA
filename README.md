# Task-specific Inconsistency Alignment for Domain Adaptive Object Detection (CVPR 2022)

Liang Zhao, Limin Wang*

This repository is the official PyTorch implementation of paper Task-specific Inconsistency Alignment for Domain Adaptive Object Detection. (The work has been accepted by CVPR2022)

The code is still being refined, so stay tuned!
<!-- ## Main requirements
torch == 1.0.0\
torchvision == 0.2.0\
Python 3\
[faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0)  -->

## Datasets
### Datasets Preparation
* **Cityscape and FoggyCityscape:** Download the [Cityscape](https://www.cityscapes-dataset.com/) dataset, see dataset preparation code in [DA-Faster RCNN](https://github.com/yuhuayc/da-faster-rcnn/tree/master/prepare_data).
* **PASCAL_VOC 07+12:** Please follow the [instruction](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC dataset.
* **Clipart:** Please follow the [instruction](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets) to prepare Clipart dataset.

### Datasets Format
All codes are written to fit for the **format of PASCAL_VOC**.  
If you want to use this code on your own dataset, please arrange the dataset in the format of PASCAL, make dataset class in ```lib/datasets/```, and add it to ```lib/datasets/factory.py```, ```lib/datasets/config_dataset.py```. Then, add the dataset option to ```lib/model/utils/parser_func.py```.

### Data Interpolation
You should use [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to generate the interpolation samples for both source and target domain, and then train the model with original and generated data.

## Models
### Pre-trained Models
In our experiments, we used two pre-trained models on ImageNet, i.e., VGG16 and ResNet101. Please download these two models from:
* **VGG16:** [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

* **ResNet101:** [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

Download them and write the path in **__C.VGG_PATH** and **__C.RESNET_PATH** at ```lib/model/utils/config.py```.

## Citation
Please cite the following reference if you utilize this repository for your project.
```
@inproceedings{tiadaod22cvpr,
  author    = {Liang Zhao and
               Limin Wang},
  title     = {Task-specific Inconsistency Alignment for Domain Adaptive Object Detection},
  booktitle = {{CVPR}},
  year      = {2022}
}
```
```
@article{zhao2022task,
  author    = {Zhao, Liang and 
               Wang, Limin},
  title     = {Task-specific Inconsistency Alignment for Domain Adaptive Object Detection},
  journal   = {arXiv preprint arXiv:2203.15345},
  year      = {2022}
}
```
