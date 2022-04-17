# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
from torch.autograd import Variable
from model.utils.net_utils import weights_init_normal


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class _ImageLayer1(nn.Module):
    def __init__(self):
        super(_ImageLayer1, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                #m.bias.data.zero_()
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.conv2, 0, 0.01)
        normal_init(self.conv3, 0, 0.01)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
        return F.sigmoid(x), feat


class _ImageLayer2(nn.Module):
    def __init__(self):
        super(_ImageLayer2, self).__init__()
        self.conv1 = conv3x3(512, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))), training=self.training)
        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.view(-1, 128)
        return self.fc(x), x


class _ImageLayer3(nn.Module):
    def __init__(self):
        super(_ImageLayer3, self).__init__()
        self.conv1 = conv3x3(512, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))), training=self.training)
        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.view(-1, 128)
        return self.fc(x), x


class _InstanceLayer(nn.Module):
    def __init__(self, dim=512):
        super(_InstanceLayer, self).__init__()
        self.conv1 = nn.Conv2d(dim, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.view(-1, 128)
        return self.fc(x)


class vgg16(_fasterRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False, context=True, num_aux1=2, num_aux2=4):
        self.model_path = "../insda/pretrained_models/vgg16_caffe.pth"
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.context = context
        self.num_aux1 = num_aux1
        self.num_aux2 = num_aux2

        _fasterRCNN.__init__(self, classes, class_agnostic, context, num_aux1, num_aux2)

    def _init_modules(self):
        vgg = models.vgg16()
        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict(
                {k: v for k, v in state_dict.items() if k in vgg.state_dict()}
            )

        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        # not using the last maxpool layer
        self.RCNN_base1 = nn.Sequential(*list(vgg.features._modules.values())[:14])
        self.RCNN_base2 = nn.Sequential(*list(vgg.features._modules.values())[14:21])
        self.RCNN_base3 = nn.Sequential(*list(vgg.features._modules.values())[21:-1])

        self.DA_img1 = _ImageLayer1()
        self.DA_img2 = _ImageLayer2()
        self.DA_img3 = _ImageLayer3()
        self.DA_Inst = _InstanceLayer(self.dout_base_model)
        
        # Fix the layers before conv3:
        for layer in range(10):
            for p in self.RCNN_base1[layer].parameters():
                p.requires_grad = False

        # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

        self.RCNN_top = vgg.classifier

        feat_d = 4096
        if self.context:
            feat_d += 128 * 3

        # not using the last maxpool layer
        self.RCNN_cls_score = nn.Linear(feat_d, self.n_classes)
        self.RCNN_cls_score_aux = nn.Linear(feat_d, self.n_classes * self.num_aux1)
        self.RCNN_bbox_pred_aux = nn.Linear(feat_d, 4 * self.num_aux2)

        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4 * self.n_classes)

    def _head_to_tail(self, pool5):
        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)

        return fc7