# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import os
import pdb
import pprint
import sys
import time
import math

import _init_paths
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from model.faster_rcnn.resnet      import resnet
from model.faster_rcnn.vgg16       import vgg16
from model.utils.config            import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils         import adjust_learning_rate, clip_gradient, create_logger, load_net, save_checkpoint, save_net
from model.utils.losses 	       import CrossEntropyLoss, FocalLoss
from parse                         import parse_args
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.roidb          import combined_roidb
from torch.autograd                import Variable
from torch.utils.data.sampler      import Sampler


class sampler(Sampler):
	def __init__(self, train_size, batch_size):
		self.num_data = train_size
		self.num_per_batch = int(train_size / batch_size)
		self.batch_size = batch_size
		self.range = torch.arange(0, batch_size).view(1, batch_size).long()
		self.leftover_flag = False
		if train_size % batch_size:
			self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
			self.leftover_flag = True

	def __iter__(self):
		rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
		self.rand_num = rand_num.expand(
			self.num_per_batch, self.batch_size) + self.range

		self.rand_num_view = self.rand_num.view(-1)

		if self.leftover_flag:
			self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

		return iter(self.rand_num_view)

	def __len__(self):
		return self.num_data


if __name__ == '__main__':
	# torch.set_num_threads(1)
	args = parse_args()

	logger, output_dir = create_logger(args)

	logger.info('Called with args:')
	logger.info(args)

	if args.dataset == "cityscape":
		args.s_imdb_name = "cityscape_2007_train_aug"
		args.t_imdb_name = "cityscape_2007_train_t"
		args.t_imdbtest_name = "cityscape_2007_test"
		args.set_cfgs = [
			"ANCHOR_SCALES",
			"[8,16,32]",
			"ANCHOR_RATIOS",
			"[0.5,1,2]",
			"MAX_NUM_GT_BOXES",
			"30",
		]
	elif args.dataset == "clipart":
		args.s_imdb_name = "voc_2007_trainval_aug+voc_2012_trainval_aug"
		args.t_imdb_name = "clipart_trainval"
		args.t_imdbtest_name = "clipart_trainval"
		args.set_cfgs = [
			"ANCHOR_SCALES",
			"[8,16,32]",
			"ANCHOR_RATIOS",
			"[0.5,1,2]",
			"MAX_NUM_GT_BOXES",
			"20",
		]
	elif args.dataset == "kitti2cityscape":
		args.s_imdb_name = "kitti_car_2012_train_aug"
		args.t_imdb_name = "daf_cityscape_car_2007_train_s"
		args.set_cfgs = [
			"ANCHOR_SCALES",
			"[8,16,32]",
			"ANCHOR_RATIOS",
			"[0.5,1,2]",
			"MAX_NUM_GT_BOXES",
			"20",
		]
	else:
		logger.info('Undefined Dataset')

	args.cfg_file = (
		"cfgs/{}_ls.yml".format(args.net)
		if args.large_scale
		else "cfgs/{}.yml".format(args.net)
	)

	if args.cfg_file is not None:
		cfg_from_file(args.cfg_file)
	if args.set_cfgs is not None:
		cfg_from_list(args.set_cfgs)

	logger.info('Using config:')
	logger.info(pprint.pformat(cfg))
	np.random.seed(cfg.RNG_SEED)

	# torch.backends.cudnn.benchmark = True
	if torch.cuda.is_available() and not args.cuda:
		logger.info("WARNING: You have a CUDA device, so you should probably run with --cuda")

	# train set
	# -- Note: Use validation set and disable the flipped to enable faster loading.
	cfg.TRAIN.USE_FLIPPED = True
	cfg.USE_GPU_NMS = args.cuda

	s_imdb, s_roidb, s_ratio_list, s_ratio_index = combined_roidb(args.s_imdb_name)
	t_imdb, t_roidb, t_ratio_list, t_ratio_index = combined_roidb(args.t_imdb_name, training=False)
	s_train_size = len(s_roidb)
	t_train_size = len(t_roidb)
	logger.info(f'source {s_train_size} & target {t_train_size} roidb entries')

	s_dataset = roibatchLoader(
		s_roidb,
		s_ratio_list,
		s_ratio_index,
		args.batch_size,
		s_imdb.num_classes,
		training=True,
	)

	s_dataloader = torch.utils.data.DataLoader(
		s_dataset,
		batch_size=args.batch_size,
		sampler=sampler(s_train_size, args.batch_size),
		num_workers=args.num_workers,
	)

	t_dataset = roibatchLoader(
		t_roidb,
		t_ratio_list,
		t_ratio_index,
		args.batch_size,
		t_imdb.num_classes,
		training=True,
	)

	t_dataloader = torch.utils.data.DataLoader(
		t_dataset,
		batch_size=args.batch_size,
		sampler=sampler(t_train_size, args.batch_size), 
		num_workers=args.num_workers,
	)

	# initilize the tensor holder here.
	im_data = torch.FloatTensor(1)
	im_info = torch.FloatTensor(1)
	num_boxes = torch.LongTensor(1)
	gt_boxes = torch.FloatTensor(1)
	tgt_im_data = torch.FloatTensor(1)
	tgt_im_info = torch.FloatTensor(1)
	tgt_num_boxes = torch.LongTensor(1)
	tgt_gt_boxes = torch.FloatTensor(1)

	# ship to cuda
	if args.cuda:
		im_data = im_data.cuda()
		im_info = im_info.cuda()
		num_boxes = num_boxes.cuda()
		gt_boxes = gt_boxes.cuda()
		tgt_im_data = tgt_im_data.cuda()
		tgt_im_info = tgt_im_info.cuda()
		tgt_num_boxes = tgt_num_boxes.cuda()
		tgt_gt_boxes = tgt_gt_boxes.cuda()

	# make variable
	im_data = Variable(im_data)
	im_info = Variable(im_info)
	num_boxes = Variable(num_boxes)
	gt_boxes = Variable(gt_boxes)
	tgt_im_data = Variable(tgt_im_data)
	tgt_im_info = Variable(tgt_im_info)
	tgt_num_boxes = Variable(tgt_num_boxes)
	tgt_gt_boxes = Variable(tgt_gt_boxes)

	if args.cuda:
		cfg.CUDA = True

	# initilize the network here.
	if args.net == 'vgg16':
		fasterRCNN = vgg16(
			s_imdb.classes, 
			pretrained=True,
			class_agnostic=args.class_agnostic,
			context=args.context, db=args.db, init=args.init,
			num_aux1=args.num_aux1, num_aux2=args.num_aux2)
	elif args.net == 'res101':
		fasterRCNN = resnet(s_imdb.classes, 101, pretrained=True,
							class_agnostic=args.class_agnostic,
							context=args.context, 
							num_aux1=args.num_aux1, num_aux2=args.num_aux2)
	else:
		logger.info('Undefined Network')

	fasterRCNN.create_architecture()

	logger.info('Number of trainable parameters: {:.3f} M'.format(sum(p.numel() for p in fasterRCNN.parameters() if p.requires_grad) / 1e6))

	lr = cfg.TRAIN.LEARNING_RATE
	lr = args.lr

	params = []
	for key, value in dict(fasterRCNN.named_parameters()).items():
		if value.requires_grad:
			if "bias" in key:
				params += [
					{
						"params": [value],
						"lr": lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
						"weight_decay": cfg.TRAIN.BIAS_DECAY
						and cfg.TRAIN.WEIGHT_DECAY
						or 0,
					}
				]
			else:
				params += [
					{
						"params": [value],
						"lr": lr,
						"weight_decay": cfg.TRAIN.WEIGHT_DECAY,
					}
				]

	if args.optimizer == "adam":
		lr = lr * 0.1
		optimizer = torch.optim.Adam(params)

	elif args.optimizer == "sgd":
		optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

	if args.resume:
		load_name = os.path.join(args.model_name)
		logger.info("loading checkpoint %s" % (load_name))
		checkpoint = torch.load(load_name)
		args.session = checkpoint['session']
		args.start_epoch = checkpoint['epoch']
		fasterRCNN.load_state_dict(checkpoint['model'])
		logger.info("loaded checkpoint %s" % (load_name))

	if args.cuda:
		fasterRCNN.cuda()

	iters_per_epoch = int(10000  / args.batch_size)
	loss_dict = {'loss': 0, 'sv': 0, 'da_img': 0, 'da_ins': 0, 'da_cls': 0, 'da_loc': 0, 'norm': 0}
	CE = CrossEntropyLoss(num_classes=2)
	FL = FocalLoss(num_classes=2, gamma=args.gamma)
	Align = CrossEntropyLoss(num_classes=2) if args.net == "vgg16" else FocalLoss(num_classes=2, gamma=args.gamma)

	def _reset_dict(loss_dict):
		for k in loss_dict.keys():
			loss_dict[k] = 0
		return loss_dict

	def _div_dict(loss_dict, step):
		for k in loss_dict.keys():
			loss_dict[k] /= step
		return loss_dict

	for epoch in range(args.start_epoch, args.max_epochs + 1):
		fasterRCNN.train()

		loss_dict = _reset_dict(loss_dict)
		start = time.time()
		if epoch > 1 and (epoch - 1) % args.lr_decay_step == 0:
			adjust_learning_rate(optimizer, args.lr_decay_gamma)
			lr *= args.lr_decay_gamma

		data_iter_s = iter(s_dataloader)
		data_iter_t = iter(t_dataloader)
		for step in range(iters_per_epoch):
			try:
				data = next(data_iter_s)
			except:
				data_iter_s = iter(s_dataloader)
				data = next(data_iter_s)
			try:
				tgt_data = next(data_iter_t)
			except:
				data_iter_t = iter(t_dataloader)
				tgt_data = next(data_iter_t)

			with torch.no_grad():
				im_data.resize_(data[0].size()).copy_(data[0])
				im_info.resize_(data[1].size()).copy_(data[1])
				gt_boxes.resize_(data[2].size()).copy_(data[2])
				num_boxes.resize_(data[3].size()).copy_(data[3])
				tgt_im_data.resize_(tgt_data[0].size()).copy_(tgt_data[0])
				tgt_im_info.resize_(tgt_data[1].size()).copy_(tgt_data[1])
				tgt_gt_boxes.resize_(1, 1, 5).zero_()
				tgt_num_boxes.resize_(1).zero_()

			if args.warmup:
				ratio =  2 / (1 + math.exp(-1 * 10 * (step + (epoch - 1) * iters_per_epoch) / (7 * iters_per_epoch))) - 1
				alpha1 = args.alpha1 * ratio
				alpha2 = args.alpha2 * ratio
				alpha3 = args.alpha3 * ratio
			else:
				alpha1 = args.alpha1
				alpha2 = args.alpha2
				alpha3 = args.alpha3

			(
				rois,
				cls_prob,
				bbox_pred,
				rpn_loss_cls,
				rpn_loss_box,
				RCNN_loss_cls,
				RCNN_loss_bbox,
				rois_label,
				img_feat1, 
				img_feat2, 
				img_feat3, 
				inst_feat,
				da_loss_cls,
				da_loss_loc
			) = fasterRCNN(
				im_data, 
				im_info, 
				gt_boxes, 
				num_boxes,
				alpha1=alpha1,
				alpha2=alpha2,
				alpha3=alpha3
			)

			(
				tgt_img_feat1, 
				tgt_img_feat2, 
				tgt_img_feat3, 
				tgt_inst_feat,
				tgt_da_loss_cls,
				tgt_da_loss_loc
			) = fasterRCNN(
				tgt_im_data, 
				tgt_im_info, 
				tgt_gt_boxes, 
				tgt_num_boxes,
				target=True,
				alpha1=alpha1,
				alpha2=alpha2,
				alpha3=alpha3
			)

			sv_loss = (
				rpn_loss_cls.mean()
				+ rpn_loss_box.mean()
				+ RCNN_loss_cls
				+ RCNN_loss_bbox
			)

			da_img_loss = 0.5 * (
				torch.mean(img_feat1 ** 2) + 
				torch.mean((1 - tgt_img_feat1) ** 2) + 
				CE(img_feat2, domain=0) * 0.15 +
				CE(tgt_img_feat2, domain=1) * 0.15 +
				FL(img_feat3, domain=0) + 
				FL(tgt_img_feat3, domain=1)
			)

			da_ins_loss = args.lamda1 * 0.5 * (
				Align(inst_feat, domain=0) +
				Align(tgt_inst_feat, domain=1)
			)

			da_cls_loss = args.lamda2 * (da_loss_cls - tgt_da_loss_cls)
			da_loc_loss = args.lamda3 * (da_loss_loc - tgt_da_loss_loc)

			loss = sv_loss + da_img_loss + da_ins_loss + da_cls_loss + da_loc_loss

			loss_dict['sv'] += sv_loss.item()
			loss_dict['da_img'] += da_img_loss.item()
			loss_dict['da_ins'] += da_ins_loss.item()
			loss_dict['da_cls'] += da_cls_loss.item()
			loss_dict['da_loc'] += da_loc_loss.item()
			loss_dict['loss'] += loss.item()

			optimizer.zero_grad()
			loss.backward()
			if args.net == "vgg16":
				loss_dict['norm'] += clip_gradient(fasterRCNN, 35.)
			optimizer.step()

			if step % args.disp_interval == 0:
				end = time.time()
				if step > 0:
					loss_dict = _div_dict(loss_dict, args.disp_interval + 1)

				out = (
					f"[epoch {epoch:2d}]"
					f"[iter {step:4d}/{iters_per_epoch:4d}] "
					f"lr: {lr:.2e}, "
					f"alpha1: {alpha1:.3f}, "
					f"alpha2: {alpha2:.3f}, "
					f"alpha3: {alpha3:.3f}, "
					f"time cost: {end-start:.4f}"
				)
				for k, v in loss_dict.items():
					out += f", {k}: {v:.10f}" if ('cls' in k or 'loc' in k) else f", {k}: {v:.4f}"
				logger.info(out)

				loss_dict = _reset_dict(loss_dict)
				start = time.time()

		if epoch > 4 or args.preserve:
			save_name = os.path.join(
				output_dir, 
				f'da_faster_rcnn_{args.dataset}_{epoch}.pth'
			)
			save_checkpoint(
				{
					'session': args.session,
					'epoch': epoch + 1,
					'model': fasterRCNN.state_dict(),
					'optimizer': optimizer.state_dict(),
					'pooling_mode': cfg.POOLING_MODE,
					'class_agnostic': args.class_agnostic,
				}, 
				save_name
			)
			logger.info(f'Model Saved as {save_name}')