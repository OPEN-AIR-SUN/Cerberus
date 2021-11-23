#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import OrderedDict
import argparse
import json
import logging
import math
import os
# import pdb
from os.path import exists, join, split
import threading
from datetime import datetime

import time

import numpy as np
import shutil

import sys
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.nn.modules import transformer
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from min_norm_solvers import MinNormSolver

import drn
import data_transforms as transforms
from model.models import DPTSegmentationModel, DPTSegmentationModelMultiHead, TransferNet, CerberusSegmentationModelMultiHead
from model.transforms import PrepareForNet


try:
    from modules import batchnormsync
except ImportError:
    pass


FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT, filename='./'+ datetime.now().strftime("%Y%m%d_%H%M%S") + '.txt')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TASK =None  # 'ATTRIBUTE', 'AFFORDANCE', 'SEGMENTATION' 
TRANSFER_FROM_TASK = None  #'ATTRIBUTE', 'AFFORDANCE', 'SEGMENTATION', or None to unable transfer


CITYSCAPE_PALETTE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)

NYU40_PALETTE = np.asarray([
    [0, 0, 0], 
    [0, 0, 80], 
    [0, 0, 160], 
    [0, 0, 240], 
    [0, 80, 0], 
    [0, 80, 80], 
    [0, 80, 160], 
    [0, 80, 240], 
    [0, 160, 0], 
    [0, 160, 80], 
    [0, 160, 160], 
    [0, 160, 240], 
    [0, 240, 0], 
    [0, 240, 80], 
    [0, 240, 160], 
    [0, 240, 240], 
    [80, 0, 0], 
    [80, 0, 80], 
    [80, 0, 160], 
    [80, 0, 240], 
    [80, 80, 0], 
    [80, 80, 80], 
    [80, 80, 160], 
    [80, 80, 240], 
    [80, 160, 0], 
    [80, 160, 80], 
    [80, 160, 160], 
    [80, 160, 240], [80, 240, 0], [80, 240, 80], [80, 240, 160], [80, 240, 240], 
    [160, 0, 0], [160, 0, 80], [160, 0, 160], [160, 0, 240], [160, 80, 0], 
    [160, 80, 80], [160, 80, 160], [160, 80, 240]], dtype=np.uint8)


AFFORDANCE_PALETTE = np.asarray([
    [0, 0, 0],
    [255, 255, 255]], dtype=np.uint8)


task_list = None
middle_task_list = None

if TASK == 'ATTRIBUTE':
    task_list = ['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny']
    FILE_DESCRIPTION = '_attribute'
    PALETTE = AFFORDANCE_PALETTE
    EVAL_METHOD = 'mIoU'
elif TASK == 'AFFORDANCE':
    task_list = ['L','M','R','S','W']
    FILE_DESCRIPTION = '_affordance'
    PALETTE = AFFORDANCE_PALETTE
    EVAL_METHOD = 'mIoU'
elif TASK =='SEGMENTATION':
    task_list = ['Segmentation']
    FILE_DESCRIPTION = ''
    PALETTE = NYU40_PALETTE
    EVAL_METHOD = 'mIoUAll'
else:
    task_list = None
    FILE_DESCRIPTION = ''
    PALETTE = None
    EVAL_METHOD = None

if TRANSFER_FROM_TASK == 'ATTRIBUTE':
    middle_task_list = ['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny']
elif TRANSFER_FROM_TASK == 'AFFORDANCE':
    middle_task_list = ['L','M','R','S','W']
elif TRANSFER_FROM_TASK =='SEGMENTATION':
    middle_task_list = ['Segmentation']
elif TRANSFER_FROM_TASK is None:
    pass


if TRANSFER_FROM_TASK is not None:
    TENSORBOARD_WRITER = SummaryWriter(comment='From_'+TRANSFER_FROM_TASK+'_TO_'+TASK)
elif TASK is not None:
    TENSORBOARD_WRITER = SummaryWriter(comment=TASK)
else:
    TENSORBOARD_WRITER = SummaryWriter(comment='Nontype')

def downsampling(x, size=None, scale=None, mode='nearest'):
    if size is None:
        size = (int(scale * x.size(2)) , int(scale * x.size(3)))
    h = torch.arange(0,size[0]) / (size[0] - 1) * 2 - 1
    w = torch.arange(0,size[1]) / (size[1] - 1) * 2 - 1
    grid = torch.zeros(size[0] , size[1] , 2)
    grid[: , : , 0] = w.unsqueeze(0).repeat(size[0] , 1)
    grid[: , : , 1] = h.unsqueeze(0).repeat(size[1] , 1).transpose(0 , 1)
    grid = grid.unsqueeze(0).repeat(x.size(0),1,1,1)
    if x.is_cuda:
        grid = grid.cuda()
    return torch.nn.functional.grid_sample(x , grid , mode = mode)

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class SegList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        data = np.array(data[0])
        if len(data.shape) == 2:
            data = np.stack([data , data , data] , axis = 2)
        data = [Image.fromarray(data)]
        if self.label_list is not None:
       	    data.append(Image.open(join(self.data_dir, self.label_list[index])))
        data = list(self.transforms(*data))
        if self.out_name:
            if self.label_list is None:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_images.txt')
        label_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)

class ConcatSegList(torch.utils.data.Dataset):
    def __init__(self, at, af, seg):
        self.at = at
        self.af = af
        self.seg = seg

    def __getitem__(self, index):
        return (self.at[index], self.af[index], self.seg[index])
    
    def __len__(self):
        return len(self.at)

class SegMultiHeadList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        data = np.array(data[0])
        if len(data.shape) == 2:
            data = np.stack([data , data , data] , axis = 2)
        data = [Image.fromarray(data)]
        
        label_data = list()
        if self.label_list is not None:
            for it in self.label_list[index].split(','):
       	        label_data.append(Image.open(join(self.data_dir, it)))
            data.append(label_data)
        data = list(self.transforms(*data))
        if self.out_name:
            if self.label_list is None:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_images.txt')
        label_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)


class SegListMS(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, scales, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()
        self.scales = scales

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        w, h = 640, 480
        data = np.array(data[0])
        if len(data.shape) == 2:
            data = np.stack([data , data , data] , axis = 2)
        data = [Image.fromarray(data)]
        if self.label_list is not None:
            data.append(Image.open(join(self.data_dir, self.label_list[index])))
        out_data = list(self.transforms(*data))
        ms_images = [self.transforms(data[0].resize((round(int(w * s)/32) * 32 , round(int(h * s)/32) * 32),
                                                    Image.BICUBIC))[0]
                     for s in self.scales]
        out_data.append(self.image_list[index])
        out_data.extend(ms_images)
        return tuple(out_data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_images.txt')
        label_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)

class SegListMSMultiHead(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, scales, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()
        self.scales = scales

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        w, h = 640, 480
        data = np.array(data[0])
        if len(data.shape) == 2:
            data = np.stack([data , data , data] , axis = 2)
        data = [Image.fromarray(data)]
        label_data = list()
        if self.label_list is not None:
            for it in self.label_list[index].split(','):
       	        label_data.append(Image.open(join(self.data_dir, it)))
            data.append(label_data)
        out_data = list(self.transforms(*data))
        ms_images = [self.transforms(data[0].resize((round(int(w * s)/32) * 32 , round(int(h * s)/32) * 32),
                                                    Image.BICUBIC))[0]
                     for s in self.scales]
        out_data.append(self.image_list[index])
        out_data.extend(ms_images)
        return tuple(out_data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_images.txt')
        label_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)


def validate(val_loader, model, criterion, eval_score=None, print_freq=10, transfer_model=None, epoch=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_array = list()
    for it in task_list:
        losses_array.append(AverageMeter())
    score = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if transfer_model is not None:
        transfer_model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input = input.cuda()
            input_var = torch.autograd.Variable(input, volatile=True)
            
            target_var = list()
            for idx in range(len(target)):
                target[idx] = target[idx].cuda(non_blocking=True)
                target_var.append(torch.autograd.Variable(target[idx], volatile=True))
            

            # compute output
            
            if transfer_model is not None:
                _, features = model(input_var)
                output = transfer_model(features)
            elif transfer_model is None:
                output, _ = model(input_var)
            softmaxf = nn.LogSoftmax()

            loss_array = list()
            for idx in range(len(output)):
                output[idx] = softmaxf(output[idx])
                loss_array.append(criterion(output[idx],target_var[idx]))

            loss = sum(loss_array)

            # measure accuracy and record loss

            losses.update(loss.item(), input.size(0))

            for idx, it in enumerate(task_list):
                (losses_array[idx]).update((loss_array[idx]).item(), input.size(0))

            scores_array = list()

            for idx in range(len(output)):
                scores_array.append(eval_score(output[idx], target_var[idx]))
            
            score.update(np.nanmean(scores_array), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {score.val:.3f} ({score.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                score=score))
            
    TENSORBOARD_WRITER.add_scalar('val_loss_average', losses.avg, global_step=epoch)
    TENSORBOARD_WRITER.add_scalar('val_score_average', score.avg, global_step=epoch)

    logger.info(' * Score {top1.avg:.3f}'.format(top1=score))

    return score.avg


def validate_cerberus(val_loader, model, criterion, eval_score=None, print_freq=10, transfer_model=None, epoch=None):
    
    task_list_array = [['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny'],
                       ['L','M','R','S','W'],
                       ['Segmentation']] 
    
    batch_time_list = list()
    losses_list = list()
    losses_array_list = list()
    score_list = list()
    score = AverageMeter()

    for i in range(3):
        batch_time_list.append(AverageMeter())
        losses_list.append(AverageMeter())
        losses_array = list()
        for it in task_list_array[i]:
            losses_array.append(AverageMeter())
        losses_array_list.append(losses_array)
        score_list.append(AverageMeter())

    # switch to evaluate mode
    model.eval()
    # if transfer_model is not None:
    #     transfer_model.eval()

    end = time.time()
    for i, pairs in enumerate(val_loader):
        for index, (input,target) in enumerate(pairs):
            with torch.no_grad():
                input = input.cuda()
                input_var = torch.autograd.Variable(input, volatile=True)
                
                target_var = list()
                for idx in range(len(target)):
                    target[idx] = target[idx].cuda(non_blocking=True)
                    target_var.append(torch.autograd.Variable(target[idx], volatile=True))
                

                # compute output
                output, _, _ = model(input_var, index)
                softmaxf = nn.LogSoftmax()

                loss_array = list()
                for idx in range(len(output)):
                    output[idx]= softmaxf(output[idx])
                    loss_array.append(criterion(output[idx],target_var[idx]))

                loss = sum(loss_array)

                # measure accuracy and record loss

                losses_list[index].update(loss.item(), input.size(0))

                for idx, it in enumerate(task_list_array[index]):
                    (losses_array_list[index][idx]).update((loss_array[idx]).item(), input.size(0))

                scores_array = list()

                if index < 2:
                    for idx in range(len(output)):
                        scores_array.append(eval_score(output[idx], target_var[idx]))
                elif index == 2:
                    for idx in range(len(output)):
                        scores_array.append(mIoUAll(output[idx], target_var[idx]))
                else:
                    assert 0 == 1
                
                tmp = np.nanmean(scores_array)
                if not np.isnan(tmp):
                    score_list[index].update(tmp, input.size(0))
                else:
                    pass

            # measure elapsed time
            batch_time_list[index].update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Score {score.val:.3f} ({score.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time_list[index], loss=losses_list[index],
                    score=score_list[index]))
        score.update(np.nanmean([score_list[0].val, score_list[1].val, score_list[2].val]))
        if i % print_freq == 0:
            logger.info('total score is:{score.val:.3f} ({score.avg:.3f})'.format(
                score = score
            ))
    
    for idx, item in enumerate(['attribute','affordance','segmentation']):
        TENSORBOARD_WRITER.add_scalar('val_'+ item +'_loss_average', losses_list[idx].avg, global_step=epoch)
        TENSORBOARD_WRITER.add_scalar('val_'+ item +'_score_average', score_list[idx].avg, global_step=epoch)

    logger.info(' * Score {top1.avg:.3f}'.format(top1=score))
    TENSORBOARD_WRITER.add_scalar('val_score_average', score.avg, global_step=epoch)


    return score.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != 255]
    correct = correct.view(-1)
    try:
        score = correct.float().sum(0).mul(100.0 / correct.size(0))
        return score.item()
    except:
        return 0

def mIoU(output, target):
    """Computes the iou for the specified values of k"""
    num_classes = output.shape[1]
    hist = np.zeros((num_classes, num_classes))
    _, pred = output.max(1)
    pred = pred.cpu().data.numpy()
    target = target.cpu().data.numpy()
    hist += fast_hist(pred.flatten(), target.flatten(), num_classes)
    ious = per_class_iu(hist) * 100
    return round(np.nanmean(ious[1]), 2)

def mIoUAll(output, target):
    """Computes the iou for the specified values of k"""
    num_classes = output.shape[1]
    hist = np.zeros((num_classes, num_classes))
    _, pred = output.max(1)
    pred = pred.cpu().data.numpy()
    target = target.cpu().data.numpy()
    hist += fast_hist(pred.flatten(), target.flatten(), num_classes)
    ious = per_class_iu(hist) * 100
    return round(np.nanmean(ious), 2)
    

def train(train_loader, model, criterion, optimizer, epoch,
          eval_score=None, print_freq=1, transfer_model=None, transfer_optim=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_array = list()
    for it in task_list:
        losses_array.append(AverageMeter())
    scores = AverageMeter()

    # switch to train mode
    model.train()

    if transfer_model is not None:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        transfer_model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        input_var = torch.autograd.Variable(input)

        target_var = list()
        for idx in range(len(target)):
            target[idx] = target[idx].cuda()
            target_var.append(torch.autograd.Variable(target[idx]))

        # compute output
        if transfer_model is None:
            output, _ = model(input_var)
        elif transfer_model is not None:
            _, features = model(input_var)
            output = transfer_model(features)

        softmaxf = nn.LogSoftmax()
        loss_array = list()

        assert len(output) == len(target)

        for idx in range(len(output)):
            output[idx] = softmaxf(output[idx])
            loss_array.append(criterion(output[idx],target_var[idx]))

        loss = sum(loss_array)

        # measure accuracy and record loss

        losses.update(loss.item(), input.size(0))

        for idx, it in enumerate(task_list):
            (losses_array[idx]).update((loss_array[idx]).item(), input.size(0))

        scores_array = list()

        for idx in range(len(output)):
            scores_array.append(eval_score(output[idx], target_var[idx]))
        
        scores.update(np.nanmean(scores_array), input.size(0))

        # compute gradient and do SGD step
        if transfer_optim is not None:
            transfer_optim.zero_grad()
        elif transfer_optim is None:
            optimizer.zero_grad()

        loss.backward()

        if transfer_optim is not None:
            transfer_optim.step()
        elif transfer_optim is None:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            losses_info = ''
            for idx, it in enumerate(task_list):
                losses_info += 'Loss_{0} {loss.val:.4f} ({loss.avg:.4f})\t'.format(it, loss=losses_array[idx])
                TENSORBOARD_WRITER.add_scalar('train_task_' + it + '_loss_val', losses_array[idx].val, 
                    global_step= epoch * len(train_loader) + i)
                TENSORBOARD_WRITER.add_scalar('train_task_' + it + '_loss_average', losses_array[idx].avg,
                    global_step= epoch * len(train_loader) + i)

            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        '{loss_info}'
                        'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,loss_info=losses_info,
                top1=scores))
            
            TENSORBOARD_WRITER.add_scalar('train_loss_val', losses.val, global_step= epoch * len(train_loader) + i)
            TENSORBOARD_WRITER.add_scalar('train_loss_average', losses.avg, global_step= epoch * len(train_loader) + i)
            TENSORBOARD_WRITER.add_scalar('train_scores_val', scores.val, global_step= epoch * len(train_loader) + i)
            TENSORBOARD_WRITER.add_scalar('train_scores_val', scores.avg, global_step= epoch * len(train_loader) + i)

    TENSORBOARD_WRITER.add_scalar('train_epoch_loss_average', losses.avg, global_step= epoch)
    TENSORBOARD_WRITER.add_scalar('train_epochscores_val', scores.avg, global_step= epoch)




def train_cerberus(train_loader, model, criterion, optimizer, epoch,
          eval_score=None, print_freq=1): # transfer_model=None, transfer_optim=None):
    
    task_list_array = [['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny'],
                       ['L','M','R','S','W'],
                       ['Segmentation']]

    root_task_list_array = ['At', 'Af', 'Seg']

    batch_time_list = list()
    data_time_list = list()
    losses_list = list()
    losses_array_list = list()
    scores_list = list()
    
    for i in range(3):
        batch_time_list.append(AverageMeter())
        data_time_list.append(AverageMeter())
        losses_list.append(AverageMeter())
        losses_array = list()
        for it in task_list_array[i]:
            losses_array.append(AverageMeter())
        losses_array_list.append(losses_array)
        scores_list.append(AverageMeter())

    model.train()

    end = time.time()

    moo = True

    for i, in_tar_name_pair in enumerate(train_loader):
        if moo :
            grads = {}
        task_loss_array = []
        for index, (input, target, name) in enumerate(in_tar_name_pair):
            # measure data loading time
            data_time_list[index].update(time.time() - end)

            if moo:

                input = input.cuda()
                input_var = torch.autograd.Variable(input)

                target_var = list()
                for idx in range(len(target)):
                    target[idx] = target[idx].cuda()
                    target_var.append(torch.autograd.Variable(target[idx]))

                # compute output
                output, _, _ = model(input_var, index)
                
                # if transfer_model is not None:
                #     output = transfer_model(output)
                softmaxf = nn.LogSoftmax()
                loss_array = list()

                assert len(output) == len(target)

                for idx in range(len(output)):
                    output[idx] = softmaxf(output[idx])
                    loss_raw = criterion(output[idx],target_var[idx])
                    
                    loss_enhance = loss_raw 

                    if torch.isnan(loss_enhance):
                        print("nan")
                        logger.info('loss_raw is: {0}'.format(loss_raw))
                        logger.info('loss_enhance is: {0}'.format(loss_enhance))
                        exit(0)
                        # loss_array.append(loss_enhance)
                    else:
                        loss_array.append(loss_enhance)

                    local_loss = sum(loss_array)
                    local_loss_enhance = local_loss 

                # backward for gradient calculate
                for cnt in model.pretrained.parameters():
                    cnt.grad = None
                model.scratch.layer1_rn.weight.grad = None
                model.scratch.layer2_rn.weight.grad = None
                model.scratch.layer3_rn.weight.grad = None
                model.scratch.layer4_rn.weight.grad = None

                local_loss_enhance.backward()

                grads[root_task_list_array[index]] = []
                for par_name, cnt in model.pretrained.named_parameters():
                    if cnt.grad is not None:
                        grads[root_task_list_array[index]].append(Variable(cnt.grad.data.clone(),requires_grad = False))
                grads[root_task_list_array[index]].append(Variable(model.scratch.layer1_rn.weight.grad.data.clone(), requires_grad = False))
                grads[root_task_list_array[index]].append(Variable(model.scratch.layer2_rn.weight.grad.data.clone(), requires_grad = False))
                grads[root_task_list_array[index]].append(Variable(model.scratch.layer3_rn.weight.grad.data.clone(), requires_grad = False))
                grads[root_task_list_array[index]].append(Variable(model.scratch.layer4_rn.weight.grad.data.clone(), requires_grad = False))
            else:
                pass
            if moo: 
                if torch.isnan(local_loss_enhance):
                    print("nan")
                    logger.info('loss_raw is: {0}'.format(local_loss))
                    logger.info('loss_enhance is: {0}'.format(local_loss_enhance))
                    exit(0)
                    # loss_array.append(loss_enhance)
                else:
                    task_loss_array.append(local_loss_enhance)

                # measure accuracy and record loss

                losses_list[index].update(local_loss_enhance.item(), input.size(0))

                for idx, it in enumerate(task_list_array[index]):
                    (losses_array_list[index][idx]).update((loss_array[idx]).item(), input.size(0))

                scores_array = list()

                if index < 2:
                    for idx in range(len(output)):
                        scores_array.append(eval_score(output[idx], target_var[idx]))
                elif index == 2:
                    for idx in range(len(output)):
                        scores_array.append(mIoUAll(output[idx], target_var[idx]))
                else:
                    assert 0 == 1
                

                scores_list[index].update(np.nanmean(scores_array), input.size(0))

            # compute gradient and do SGD step
            if index == 2:
                if moo:
                    del input, target, input_var, target_var
                    task_loss_array_new = []
                    for index_new, (input_new, target_new, _) in enumerate(in_tar_name_pair):
                        input_var_new = torch.autograd.Variable(input_new.cuda())
                        target_var_new = [torch.autograd.Variable(target_new[idx].cuda()) for idx in range(len(target_new))]
                        output_new, _, _ = model(input_var_new, index_new)
                        loss_array_new = [criterion(softmaxf(output_new[idx]),target_var_new[idx]) \
                            for idx in range(len(output_new))]
                        local_loss_new = sum(loss_array_new)
                        task_loss_array_new.append(local_loss_new)
                    assert len(task_loss_array_new) == 3
                    sol, min_norm = MinNormSolver.find_min_norm_element([grads[cnt] for cnt in root_task_list_array])

                    logger.info('scale is: |{0}|\t|{1}|\t|{2}|\t'.format(sol[0], sol[1], sol[2]))
                    
                    loss_new = 0
                    loss_new = sol[0] * task_loss_array_new[0] + sol[1] * task_loss_array_new[1] \
                         + sol[2] * task_loss_array_new[2]
                    
                    optimizer.zero_grad()
                    loss_new.backward()
                    optimizer.step()
                else:
                    assert len(task_loss_array) == 3

                    loss = sum(task_loss_array)
                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()
            
            if moo:
            # measure elapsed time
                batch_time_list[index].update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    losses_info = ''
                    for idx, it in enumerate(task_list_array[index]):
                        losses_info += 'Loss_{0} {loss.val:.4f} ({loss.avg:.4f}) \t'.format(it, loss=losses_array_list[index][idx])
                        TENSORBOARD_WRITER.add_scalar('train_task_'+ it +'_loss_val', losses_array_list[index][idx].val,
                            global_step= epoch * len(train_loader) + i)
                        TENSORBOARD_WRITER.add_scalar('train_task_'+ it +'_loss_avg', losses_array_list[index][idx].avg,
                            global_step= epoch * len(train_loader) + i)

                    logger.info('Epoch: [{0}][{1}/{2}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                '{loss_info}'
                                'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time_list[index],
                        data_time=data_time_list[index], loss=losses_list[index],loss_info=losses_info,
                        top1=scores_list[index]))
                    logger.info('File name is: {}'.format(','.join(name)))
                    TENSORBOARD_WRITER.add_scalar('train_'+ str(index) +'_losses_val', losses_list[index].val,
                        global_step= epoch * len(train_loader) + i)
                    TENSORBOARD_WRITER.add_scalar('train_'+ str(index) +'_losses_avg', losses_list[index].avg,
                        global_step= epoch * len(train_loader) + i)
                    TENSORBOARD_WRITER.add_scalar('train_'+ str(index) +'_score_val', scores_list[index].val,
                        global_step= epoch * len(train_loader) + i)
                    TENSORBOARD_WRITER.add_scalar('train_'+ str(index) +'_score_avg', scores_list[index].avg,
                        global_step= epoch * len(train_loader) + i)
    for i in range(3):
        TENSORBOARD_WRITER.add_scalar('train_epoch_loss_average', losses_list[index].avg, global_step= epoch)
        TENSORBOARD_WRITER.add_scalar('train_epoch_scores_val', scores_list[index].avg, global_step= epoch)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def train_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size

    print(' '.join(sys.argv))

    for k, v in args.__dict__.items():
        print(k, ':', v)
    
    if len(task_list) == 1:
        single_model = DPTSegmentationModel(args.classes, backbone="vitb_rn50_384")
    else:
        single_model = DPTSegmentationModelMultiHead(args.classes, task_list, backbone="vitb_rn50_384")
    model = single_model.cuda()

    if args.trans:
        if len(middle_task_list) == 1:
            single_model = DPTSegmentationModel(40, backbone="vitb_rn50_384")
        else:
            single_model = DPTSegmentationModelMultiHead(2, middle_task_list, backbone="vitb_rn50_384")
        model = single_model.cuda()
        model_trans = TransferNet(middle_task_list, task_list)
        model_trans = model_trans.cuda()
        
    criterion = nn.NLLLoss2d(ignore_index=255)

    criterion.cuda()

    # Data loading code
    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'],
                                     std=info['std'])
    t = []

    if args.random_rotate > 0:
        t.append(transforms.RandomRotateMultiHead(args.random_rotate))
    if args.random_scale > 0:
        t.append(transforms.RandomScaleMultiHead(args.random_scale))
    t.extend([transforms.RandomCropMultiHead(crop_size),
                transforms.RandomHorizontalFlipMultiHead(),
                transforms.ToTensorMultiHead(),
                normalize])
            
    train_loader = torch.utils.data.DataLoader(
        SegMultiHeadList(data_dir, 'train', transforms.Compose(t)),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
            SegMultiHeadList(data_dir, 'val', transforms.Compose([
            transforms.RandomCropMultiHead(crop_size),
            transforms.ToTensorMultiHead(),
            normalize,
        ])),
        batch_size=1, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )


    # define loss function (criterion) and pptimizer
    optimizer = torch.optim.SGD(single_model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.trans:
        trans_optim = torch.optim.SGD(model_trans.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                model.state_dict()[name].copy_(param)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.trans_resume:
        if os.path.isfile(args.trans_resume):
            print("=> loading trans checkpoint '{}'".format(args.trans_resume))
            checkpoint = torch.load(args.trans_resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                model_trans.state_dict()[name].copy_(param)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.trans_resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                model.state_dict()[name].copy_(param)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, criterion, eval_score=eval(EVAL_METHOD), epoch=0)
        return

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch
        if args.trans:
            train(train_loader, model, criterion, optimizer, epoch, 
              eval_score=eval(EVAL_METHOD), transfer_model=model_trans, transfer_optim=trans_optim)
        else:
            train(train_loader, model, criterion, optimizer, epoch,
              eval_score=eval(EVAL_METHOD))

        # evaluate on validation set
        if args.trans:
            prec1 = validate(val_loader, model, criterion,
              eval_score=eval(EVAL_METHOD), transfer_model=model_trans, epoch=epoch)
        else:
            prec1 = validate(val_loader, model, criterion, eval_score=eval(EVAL_METHOD), epoch=epoch)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        checkpoint_path = 'checkpoint_latest.pth.tar'
        
        if args.trans:
            checkpoint_path = str(len(middle_task_list)) + 'transfer'+ \
                str(len(task_list))+checkpoint_path
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model_trans.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=checkpoint_path)
            if (epoch + 1) % 10 == 0:
                history_path = str(len(middle_task_list)) + 'transfer'+ \
                    str(len(task_list)) + 'checkpoint_{:03d}.pth.tar'.format(epoch + 1)
                shutil.copyfile(checkpoint_path, history_path)
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=checkpoint_path)
            if (epoch + 1) % 10 == 0:
                history_path = 'checkpoint_{:03d}.pth.tar'.format(epoch + 1)
                shutil.copyfile(checkpoint_path, history_path)

def train_seg_cerberus(args):
    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size

    print(' '.join(sys.argv))

    for k, v in args.__dict__.items():
        print(k, ':', v)
    
    single_model = CerberusSegmentationModelMultiHead(backbone="vitb_rn50_384")
    model = single_model.cuda()
        
    criterion = nn.NLLLoss2d(ignore_index=255)

    criterion.cuda()

    # Data loading code
    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))


    normalize = transforms.Normalize(mean=info['mean'],
                                     std=info['std'])
    t = []

    if args.random_rotate > 0:
        t.append(transforms.RandomRotateMultiHead(args.random_rotate))
    if args.random_scale > 0:
        t.append(transforms.RandomScaleMultiHead(args.random_scale))
    t.extend([transforms.RandomCropMultiHead(crop_size),
                transforms.RandomHorizontalFlipMultiHead(),
                transforms.ToTensorMultiHead(),
                normalize])

    dataset_at_train = SegMultiHeadList(data_dir, 'train_attribute', transforms.Compose(t), out_name=True)
    dataset_af_train = SegMultiHeadList(data_dir, 'train_affordance', transforms.Compose(t), out_name=True)
    dataset_seg_train = SegMultiHeadList(data_dir, 'train', transforms.Compose(t), out_name=True)

    train_loader = (torch.utils.data.DataLoader(
        ConcatSegList(dataset_at_train, dataset_af_train, dataset_seg_train),
         batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True
    ))

    dataset_at_val = SegMultiHeadList(data_dir, 'val_attribute', transforms.Compose([
                transforms.RandomCropMultiHead(crop_size),
                transforms.ToTensorMultiHead(),
                normalize,
            ]))
    dataset_af_val = SegMultiHeadList(data_dir, 'val_affordance', transforms.Compose([
                transforms.RandomCropMultiHead(crop_size),
                transforms.ToTensorMultiHead(),
                normalize,
            ]))
    dataset_seg_val = SegMultiHeadList(data_dir, 'val', transforms.Compose([
                transforms.RandomCropMultiHead(crop_size),
                transforms.ToTensorMultiHead(),
                normalize,
            ]))

    val_loader = (torch.utils.data.DataLoader(
        ConcatSegList(dataset_at_val, dataset_af_val, dataset_seg_val),
        batch_size=1, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=True
    ))

    # define loss function (criterion) and pptimizer
    optimizer = torch.optim.SGD([
                                {'params':single_model.pretrained.parameters()},
                                {'params':single_model.scratch.parameters()}],
                                # {'params':single_model.sigma.parameters(), 'lr': args.lr * 0.01}],
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                model.state_dict()[name].copy_(param)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                if name[:5] == 'sigma':
                    model.state_dict()[name].copy_(param)
                else:
                    # model.state_dict()[name].copy_(param)
                    pass
            print("=> loaded sigma checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.pretrained_model:
        if os.path.isfile(args.pretrained_model):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrained_model))
            checkpoint = torch.load(args.pretrained_model)
            for name, param in checkpoint['state_dict'].items():
                if name[:5] == 'sigma':
                    pass
                    # model.state_dict()[name].copy_(param)
                else:
                    model.state_dict()[name].copy_(param)
                    # pass
            print("=> loaded model checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate_cerberus(val_loader, model, criterion, eval_score=mIoU, epoch=0)
        return

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))

        train_cerberus(train_loader, model, criterion, optimizer, epoch,
        eval_score=mIoU)
        
        #if epoch%10==1:
        prec1 = validate_cerberus(val_loader, model, criterion, eval_score=mIoU, epoch=epoch)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        checkpoint_path = 'checkpoint_latest.pth.tar'
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)
        if (epoch + 1) % 10 == 0:
            history_path = 'checkpoint_{:03d}.pth.tar'.format(epoch + 1)
            shutil.copyfile(checkpoint_path, history_path)


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    #adjust the learning rate of sigma
    optimizer.param_groups[-1]['lr'] = lr * 0.01
    
    return lr


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def save_colorful_images(predictions, filenames, output_dir, palettes):
   """
   Saves a given (B x C x H x W) into an image file.
   If given a mini-batch tensor, will save the tensor as a grid of images.
   """
   for ind in range(len(filenames)):
       im = Image.fromarray(palettes[predictions[ind].squeeze()])
       fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
       out_dir = split(fn)[0]
       if not exists(out_dir):
           os.makedirs(out_dir)
       im.save(fn)

def resize_4d_tensor(tensor, width, height):
    tensor_cpu = tensor.cpu().numpy()
    if tensor.size(2) == height and tensor.size(3) == width:
        return tensor_cpu
    out_size = (tensor.size(0), tensor.size(1), height, width)
    out = np.empty(out_size, dtype=np.float32)

    def resize_one(i, j):
        out[i, j] = np.array(
            Image.fromarray(tensor_cpu[i, j]).resize(
                (width, height), Image.BILINEAR))

    def resize_channel(j):
        for i in range(tensor.size(0)):
            out[i, j] = np.array(
                Image.fromarray(tensor_cpu[i, j]).resize(
                    (width, height), Image.BILINEAR))

    workers = [threading.Thread(target=resize_channel, args=(j,))
               for j in range(tensor.size(1))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return out


def test_ms(eval_data_loader, model, num_classes, scales,
            output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist_array_acc = list()
    hist_array = list()
    iou_compute_cmd = 'per_class_iu(hist_array[idx])'
    if num_classes == 2:
        iou_compute_cmd = '[' + iou_compute_cmd + '[1]]'

    iou_compute_cmd_acc = 'per_class_iu(hist_array_acc[idx])'
    if num_classes == 2:
        iou_compute_cmd_acc = '[' + iou_compute_cmd_acc + '[1]]'

    for i in range(len(task_list)):
        hist_array_acc.append(np.zeros((num_classes, num_classes)))
        hist_array.append(np.zeros((num_classes, num_classes)))

    num_scales = len(scales)
    for itera, input_data in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        
        if has_gt:
            name = input_data[2]
            label = input_data[1]
        else:
            name = input_data[1]

        logger.info('file name is %s', name)
        
        h, w = input_data[0].size()[2:4]
        images = input_data[-num_scales:]
        outputs = []

        with torch.no_grad():
            for image in images:
                image_var = Variable(image, requires_grad=False)
                image_var = image_var.cuda()
                final, _ = model(image_var)
                final_array = list()
                for entity in final:
                    final_array.append(entity.data)
                outputs.append(final_array)

            final = list()
            for label_idx in range(len(outputs[0])):
                tmp_tensor_list = list()
                for out in outputs:
                    tmp_tensor_list.append(resize_4d_tensor(out[label_idx], w, h))
                
                final.append(sum(tmp_tensor_list))
            pred = list()
            for label_entity in final:
                pred.append(label_entity.argmax(axis=1))

        batch_time.update(time.time() - end)
        if save_vis:
            for idx in range(len(label)):
                assert len(name) == 1
                file_name = (name[0][:-4] + task_list[idx] + '.png',)
                save_output_images(pred[idx], file_name, output_dir)
                save_colorful_images(pred[idx], file_name, output_dir + '_color',
                                    PALETTE)
        if has_gt:
            map_score_array = list()
            for idx in range(len(label)):
                label[idx] = label[idx].numpy()
                hist_array[idx] = fast_hist(pred[idx].flatten(), label[idx].flatten(), num_classes)
                hist_array_acc[idx] += hist_array[idx]

                map_score_array.append(round(np.nanmean(eval(iou_compute_cmd)) * 100, 2))

                logger.info('===> task${}$ mAP {mAP:.3f}'.format(
                    task_list[idx],
                    mAP= map_score_array[idx]))
            
            if len(map_score_array) > 1:
                assert len(map_score_array) == len(label)
                logger.info('===> task${}$ mAP {mAP:.3f}'.format(
                    TASK,
                    mAP= round(np.nanmean(map_score_array),2)))

        end = time.time()
        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(itera, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
    if has_gt: #val
        ious = list()
        for idx in range(len(hist_array_acc)):
            tmp_result = [i * 100.0 for i in eval(iou_compute_cmd_acc)]
            ious.append(tmp_result)
        for idx, i in enumerate(ious):
            logger.info('task %s', task_list[idx])
            logger.info(' '.join('{:.3f}'.format(ii) for ii in i))
        return round(np.nanmean(ious), 2)


def test_ms_cerberus(eval_data_loader, model, scales,
            output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    
    task_list_array = [['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny'],
                       ['L','M','R','S','W'],
                       ['Segmentation']]

    task_name = ['Attribute', 'Affordance', 'Segmentation']
    
    batch_time_array = list()
    data_time_array = list()
    hist_array_array = list()
    hist_array_array_acc = list()
    for i in range(3):
        batch_time_array.append(AverageMeter())
        data_time_array.append(AverageMeter())
        hist_array_array.append([])
        hist_array_array_acc.append([])
        if i < 2:
            num_classes = 2
        elif i == 2:
            num_classes = 40
        else:
            assert 0 == 1
        for j in range(len(task_list_array[i])):
            hist_array_array[i].append(np.zeros((num_classes, num_classes)))
            hist_array_array_acc[i].append(np.zeros((num_classes, num_classes)))



    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    end = time.time()

    for i, in_tar_pair in enumerate(zip(eval_data_loader[0], eval_data_loader[1], eval_data_loader[2])):
        for index, input in enumerate(in_tar_pair):
            if index < 2:
                num_classes = 2
                PALETTE = AFFORDANCE_PALETTE
            elif index == 2:
                num_classes = 40
                PALETTE = NYU40_PALETTE
            else:
                assert 0 == 1
            task_list = task_list_array[index]
            iou_compute_cmd = 'per_class_iu(hist_array_array[index][idx])'
            if num_classes == 2:
                iou_compute_cmd = '[' + iou_compute_cmd + '[1]]'

            num_scales = len(scales)
            
            data_time_array[index].update(time.time() - end)
        
            if has_gt:
                name = input[2]
                label = input[1]
            else:
                name = input[1]
            
            logger.info('file name is %s', name)

            h, w = input[0].size()[2:4]
            images = input[-num_scales:]
            outputs = []

            with torch.no_grad():
                for image in images:
                    image_var = Variable(image, requires_grad=False)
                    image_var = image_var.cuda()
                    final, _, _ = model(image_var, index)
                    final_array = list()
                    for entity in final:
                        final_array.append(entity.data)
                    outputs.append(final_array)

                final = list()
                for label_idx in range(len(outputs[0])):
                    tmp_tensor_list = list()
                    for out in outputs:
                        tmp_tensor_list.append(resize_4d_tensor(out[label_idx], w, h))
                    
                    final.append(sum(tmp_tensor_list))
                pred = list()
                for label_entity in final:
                    pred.append(label_entity.argmax(axis=1))

            batch_time_array[index].update(time.time() - end)
            if save_vis:
                for idx in range(len(pred)):
                    assert len(name) == 1
                    file_name = (name[0][:-4] + task_list[idx] + '.png',)
                    save_output_images(pred[idx], file_name, output_dir)
                    save_colorful_images(pred[idx], file_name, output_dir + '_color',
                                        PALETTE)
                    if index == 2:
                        gt_name = (name[0][:-4] + task_list[idx] + '_gt.png',)    
                        label_mask =  (label[idx]==255)

                        save_colorful_images((label[idx]-label_mask*255).numpy(), gt_name, output_dir + '_color',PALETTE)

            if has_gt:
                map_score_array = list()
                for idx in range(len(label)):
                    label[idx] = label[idx].numpy()
                    hist_array_array[index][idx] = fast_hist(pred[idx].flatten(),
                     label[idx].flatten(), num_classes)
                    hist_array_array_acc[index][idx] += hist_array_array[index][idx]
                    
                    map_score_array.append(round(np.nanmean([it * 100.0 for it in eval(iou_compute_cmd)]), 2))

                    logger.info('===> task${}$ mAP {mAP:.3f}'.format(
                        task_list[idx],
                        mAP=map_score_array[idx]))
                
                if len(map_score_array) > 1:
                    assert len(map_score_array) == len(label)
                    logger.info('===> task${}$ mAP {mAP:.3f}'.format(
                        task_name[index],
                        mAP=round(np.nanmean(map_score_array),2)))
                        

            end = time.time()
            logger.info('Eval: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        .format(i, len(eval_data_loader[index]), batch_time=batch_time_array[index],
                                data_time=data_time_array[index]))
    if has_gt: #val
        ious_array = list()
        for index, iter in enumerate(hist_array_array_acc):
            ious = list()
            for idx, jter in enumerate(iter):
                iou_compute_cmd = 'per_class_iu(hist_array_array_acc[index][idx])'
                if index < 2:
                    iou_compute_cmd = '[' + iou_compute_cmd + '[1]]'
                tmp_result = [i * 100.0 for i in eval(iou_compute_cmd)]
                ious.append(tmp_result)
            ious_array.append(ious)
        task_name = ['attribute', 'affordance','segmentation']
        for num, ious in enumerate(ious_array):
            for idx, i in enumerate(ious):
                logger.info('task %s', task_list_array[num][idx])
                logger.info(' '.join('{:.3f}'.format(ii) for ii in i))
        for num, ious in enumerate(ious_array):
            logger.info('task %s : %.2f',task_name[num], 
                [np.nanmean(i) for i in ious_array][num])

        return round(np.nanmean([np.nanmean(i) for i in ious_array]), 2)


def test_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase

    for k, v in args.__dict__.items():
        print(k, ':', v)

    if len(task_list) == 1:
        single_model = DPTSegmentationModel(args.classes, backbone="vitb_rn50_384")
    else:
        single_model = DPTSegmentationModelMultiHead(args.classes, task_list, backbone="vitb_rn50_384")

    checkpoint = torch.load(args.resume)
    
    for name, param in checkpoint['state_dict'].items():
        # name = name[7:]
        single_model.state_dict()[name].copy_(param)
    
    if args.pretrained:
        single_model.load_state_dict(torch.load(args.pretrained))
    model = single_model.cuda()

    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'], std=info['std'])
    scales = [0.9, 1, 1.25]

    if args.ms:
        dataset = SegListMSMultiHead(data_dir, phase, transforms.Compose([
            transforms.ToTensorMultiHead(),
            normalize,
        ]), scales)
    else:
        dataset = SegMultiHeadList(data_dir, phase, transforms.Compose([
            transforms.ToTensorMultiHead(),
            normalize,
        ]), out_name=True)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )

    cudnn.benchmark = True

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                model.state_dict()[name].copy_(param)
            # model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    out_dir = '{}_{:03d}_{}'.format(args.arch, start_epoch, phase)
    if len(args.test_suffix) > 0:
        out_dir += '_' + args.test_suffix
    if args.ms:
        out_dir += '_ms'

    if args.ms:
        mAP = test_ms(test_loader, model, args.classes, save_vis=True,
                      has_gt=phase != 'test' or args.with_gt,
                      output_dir=out_dir,
                      scales=scales)

    logger.info('%s mAP: %f', TASK, mAP)

def test_seg_cerberus(args):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase

    task_list_array = [['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny'],
                    ['L','M','R','S','W'],
                    ['Segmentation']] 

    for k, v in args.__dict__.items():
        print(k, ':', v)


    single_model = CerberusSegmentationModelMultiHead(backbone="vitb_rn50_384")

    checkpoint = torch.load(args.resume)
    
    for name, param in checkpoint['state_dict'].items():
        # name = name[7:]
        single_model.state_dict()[name].copy_(param)
    
    model = single_model.cuda()

    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'], std=info['std'])
    scales = [0.9, 1, 1.25]

    test_loader_list = []
    if args.ms:
        for i in ['_attribute', '_affordance', '']:
            test_loader_list.append(torch.utils.data.DataLoader(
                SegListMSMultiHead(data_dir, phase + i, transforms.Compose([
                    transforms.ToTensorMultiHead(),
                    normalize,]), 
                scales
                ),
                batch_size=batch_size, shuffle=False, num_workers=num_workers,
                pin_memory=False
            ))
    else:
        assert 0 == 1


    cudnn.benchmark = True

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                model.state_dict()[name].copy_(param)
            # model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    out_dir = '{}_{:03d}_{}'.format(args.arch, start_epoch, phase)
    if len(args.test_suffix) > 0:
        out_dir += '_' + args.test_suffix
    if args.ms:
        out_dir += '_ms'

    if args.ms:
        mAP = test_ms_cerberus(test_loader_list, model, save_vis=True,
                      has_gt=phase != 'test' or args.with_gt,
                      output_dir=out_dir,
                      scales=scales)
    else:
        assert 0 == 1, 'please add the argument --ms'
    if mAP is not None:
        logger.info('%s mAP: %f', 'average mAP is: ', mAP)



def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('-d', '--data-dir', default='../dataset/nyud2')
    parser.add_argument('-c', '--classes', default=0, type=int)
    parser.add_argument('-s', '--crop-size', default=0, type=int)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--trans-resume', default='', type=str, metavar='PATH',
                        help='path to latest trans checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('--pretrained-model', dest='pretrained_model',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--load-release', dest='load_rel', default=None)
    parser.add_argument('--phase', default='val')
    parser.add_argument('--random-scale', default=0, type=float)
    parser.add_argument('--random-rotate', default=0, type=int)
    parser.add_argument('--bn-sync', action='store_true')
    parser.add_argument('--ms', action='store_true',
                        help='Turn on multi-scale testing')
    parser.add_argument('--trans', action='store_true',
                        help='Turn on transfer learning')
    parser.add_argument('--with-gt', action='store_true')
    parser.add_argument('--test-suffix', default='', type=str)
    args = parser.parse_args()

    assert args.data_dir is not None
    assert args.classes > 0

    print(' '.join(sys.argv))
    print(args)

    if args.bn_sync:
        drn.BatchNorm = batchnormsync.BatchNormSync

    return args


def main():
    args = parse_args()
    if args.cmd == 'train':
        train_seg_cerberus(args)
    elif args.cmd == 'test':
        test_seg_cerberus(args)


if __name__ == '__main__':
    main()
