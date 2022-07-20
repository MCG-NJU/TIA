import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.rpn import _RPN
from model.utils.config import cfg
from model.utils.net_utils import (
    _affine_grid_gen,
    _affine_theta,
    _crop_pool_layer,
    _smooth_l1_loss,
    ReverseLayerF
)
from torch.autograd import Variable

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import 

def entropy(preds):
    H = -preds * torch.log(preds + 1e-5)
    return H.sum(dim=2)


class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic, context, num_aux1, num_aux2):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.context = context
        self.num_aux1 = num_aux1
        self.num_aux2 = num_aux2

        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

    def forward(self, 
                im_data, 
                im_info, 
                gt_boxes, 
                num_boxes,
                alpha1=1.0,
                alpha2=1.0,
                alpha3=1.0,
                target=False
        ):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat1 = self.RCNN_base1(im_data)
        base_feat2 = self.RCNN_base2(base_feat1)
        base_feat = self.RCNN_base3(base_feat2)

        img_feat1, feat1 = self.DA_img1(ReverseLayerF.apply(base_feat1, 1.0))
        img_feat2, feat2 = self.DA_img2(ReverseLayerF.apply(base_feat2, 1.0))
        img_feat3, feat3 = self.DA_img3(ReverseLayerF.apply(base_feat, 1.0))

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))
            
        inst_feat = None
        if self.training:
            inst_feat = self.DA_inst(ReverseLayerF.apply(pooled_feat, alpha1))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        if self.context:
            feat1 = feat1.view(1, -1).repeat(pooled_feat.size(0), 1)
            feat2 = feat2.view(1, -1).repeat(pooled_feat.size(0), 1)
            feat3 = feat3.view(1, -1).repeat(pooled_feat.size(0), 1)
            pooled_feat = torch.cat((feat1, feat2, feat3, pooled_feat), 1)

        da_loss_cls, da_loss_loc = 0, 0
        if self.training:
            rev_pooled_feat = ReverseLayerF.apply(pooled_feat, alpha2)
            cls_score_aux = self.RCNN_cls_score_aux(rev_pooled_feat).view(pooled_feat.size(0), -1, self.num_aux1)
            cls_prob_aux = F.softmax(cls_score_aux, dim=1)
            weight = cls_prob_aux.mean(dim=2)
            cls_prob_aux = F.softmax(cls_prob_aux, dim=2)
            ent = entropy(cls_prob_aux)
            ent *= weight
            da_loss_cls = -ent.sum(dim=1).mean()

            rev_pooled_feat = ReverseLayerF.apply(pooled_feat, alpha3)
            bbox_pred_aux = self.RCNN_bbox_pred_aux(rev_pooled_feat).view(pooled_feat.size(0), -1, self.num_aux2)
            da_loss_loc = bbox_pred_aux.std(dim=2).mean(dim=1).mean()

        if target:
            return (
                img_feat1,
                img_feat2,
                img_feat3,
                inst_feat,
                da_loss_cls, 
                da_loss_loc
            )  

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        bbox_pred_aux = self.RCNN_bbox_pred_aux(pooled_feat.detach()).view(pooled_feat.size(0), -1, self.num_aux2)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_score_aux = self.RCNN_cls_score_aux(pooled_feat.detach()).view(pooled_feat.size(0), -1, self.num_aux1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = (
                F.cross_entropy(cls_score, rois_label).mean() +
                sum([
                    F.cross_entropy(cls_score_aux[:, :, i], rois_label).mean()
                    for i in range(self.num_aux1)
                ])
            )

            # bounding box regression L1 loss
            RCNN_loss_bbox = (
                _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws).mean() + 
                sum([
                    _smooth_l1_loss(bbox_pred_aux[:, :, i], rois_target, rois_inside_ws, rois_outside_ws).mean() 
                    for i in range(self.num_aux2)
                ])
            )

        cls_prob = F.softmax(cls_score, 1).view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return (
            rois, 
            cls_prob, 
            bbox_pred, 
            rpn_loss_cls, 
            rpn_loss_bbox, 
            RCNN_loss_cls, 
            RCNN_loss_bbox, 
            rois_label,
            img_feat1,
            img_feat2,
            img_feat3,
            inst_feat,
            da_loss_cls,
            da_loss_loc
        )

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

        # alternative initialization (might be better)
        # for i in range(self.num_aux1):
        #     self.RCNN_cls_score_aux.weight[i * self.n_classes: (i+1) * self.n_classes, :].data.normal_(0, 0.01 * (i + 1))
        #     self.RCNN_cls_score_aux.bias.data.zero_()

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
