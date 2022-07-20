import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=2):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        print(f'Softmax Cross Entropy Loss with num_classes={self.num_classes}')

    def forward(self, input, domain):
        B, C = input.shape
        probs = F.softmax(input, 1)
        label = F.one_hot(torch.tensor(domain).repeat(B).cuda(), self.num_classes).float()

        loss = -(probs.log() * label).sum(-1)
        return loss.mean(-1)


class FocalLoss(nn.Module):
    def __init__(self, num_classes=2, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        print(f'Focal Loss with gamma={self.gamma} & num_classes={self.num_classes}')

    def forward(self, input, domain):
        B, C = input.shape
        probs = F.softmax(input, 1)
        label = F.one_hot(torch.tensor(domain).repeat(B).cuda(), self.num_classes).float()

        probs = (probs * label).sum(-1)
        loss = -torch.pow(1 - probs, self.gamma) * probs.log()
        return loss.mean()
