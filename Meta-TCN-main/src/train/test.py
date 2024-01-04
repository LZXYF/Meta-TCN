import datetime


import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm
import copy

from train.utils import *
from dataset.sampler2 import SerialSampler,  task_sampler
from dataset import utils
from tools.tool import reidx_y, neg_dist

from torch import autograd
from collections import OrderedDict
import itertools
from train.ContrastFunction import SupConLoss

def test_one(task, class_names, model, optG, criterion, args, grad, cs):
    '''
        Train the model on one sampled task.
    '''
    model['G'].cuda(args.cuda)
    model['G'].eval()

    support, query = task

    YS = support['label']
    YQ = query['label']

    sampled_classes = torch.unique(support['label']).cpu().numpy().tolist()

    class_names_dict = {}
    class_names_dict['label'] = class_names['label'][sampled_classes]
    class_names_dict['text'] = class_names['text'][sampled_classes]
    class_names_dict['text_len'] = class_names['text_len'][sampled_classes]
    class_names_dict['is_support'] = False
    class_names_dict = utils.to_tensor(class_names_dict, args.cuda, exclude_keys=['is_support'])

    YS, YQ = reidx_y(args, YS, YQ)

    """维度填充"""
    if support['text'].shape[1] > class_names_dict['text'].shape[1]:
        zero = torch.zeros(
            (class_names_dict['text'].shape[0], support['text'].shape[1] - class_names_dict['text'].shape[1]),
            dtype=torch.long)
        class_names_dict['text'] = torch.cat((class_names_dict['text'], zero.cuda(args.cuda)), dim=-1)
    elif support['text'].shape[1] < class_names_dict['text'].shape[1]:
        zero = torch.zeros(
            (support['text'].shape[0], class_names_dict['text'].shape[1] - support['text'].shape[1]),
            dtype=torch.long)
        support['text'] = torch.cat((support['text'], zero.cuda(args.cuda)), dim=-1)

    pos = {"text": None}
    neg = {"text": None}
    ancher = {'text': None}
    allsample = {'text': None}
    allsample['text'] = torch.cat([support['text'], class_names_dict['text']], dim=0)
    allsample['label'] = torch.cat([support['label'], class_names_dict["label"]], dim=0)

    for i in range(len(allsample['text'])):
        tmp_pos = torch.ones(args.way - 1,
                             allsample['text'].shape[1]).cuda(args.cuda)
        indices = torch.nonzero(torch.eq(class_names_dict['label'], allsample['label'][i]))
        index = indices[0][0].item()
        tmp_pos = tmp_pos * class_names_dict['text'][index, :]

        if pos['text'] is None:
            pos['text'] = tmp_pos
        else:
            pos['text'] = torch.cat([pos['text'], tmp_pos], dim=0)

        if ancher["text"] is None:
            ancher['text'] = allsample['text'][i].repeat(args.way - 1, 1).long()
        else:
            ancher['text'] = torch.cat([ancher['text'], allsample['text'][i].repeat(args.way - 1, 1).long()], dim=0)

        for j in range(len(class_names_dict['label'])):
            if j != index:
                tmp_neg = torch.ones(1,
                                     allsample['text'].shape[1]).cuda(args.cuda)
                tmp_neg = tmp_neg * class_names_dict['text'][j, :]
                if neg['text'] is None:
                    neg['text'] = tmp_neg
                else:
                    neg['text'] = torch.cat([neg['text'], tmp_neg], dim=0)

    pos['text'] = pos['text'].long()
    neg['text'] = neg["text"].long()
    ancher['text'] = ancher['text'].long()

    ancher_feature, pos_feature, neg_feature = model['G'](ancher, pos, neg)
    class_feature = model['G'].forward2(class_names_dict)
    class_loss = cs(class_feature)

    support_feature = model['G'].forward2(support)
    support_feature = support_feature.reshape(args.way, args.shot, -1)
    support_feature = support_feature.mean(1)
    support_loss = cs.forward(support_feature)

    loss_weight = torch.cat( 
        (torch.ones([(args.way-1) * support['text'].shape[0]]), args.train_loss_weight * torch.ones([20])), 0)
    if args.cuda != -1:
        loss_weight = loss_weight.cuda(args.cuda)

    loss = criterion(ancher_feature, pos_feature, neg_feature, loss_weight=loss_weight)
    loss = loss + args.class_loss_weight * class_loss + args.proto_loss_weight * support_loss
    zero_grad(model['G'].parameters())

    grads_fc1 = autograd.grad(loss, model['G'].fc1.parameters(), allow_unused=True, retain_graph=True)
    fast_weights_fc1, orderd_params_fc1 = model['G'].cloned_fc1_dict(), OrderedDict()

    grads_fc2 = autograd.grad(loss, model['G'].fc2.parameters(), allow_unused=True, retain_graph=True)
    fast_weights_fc2, orderd_params_fc2 = model['G'].cloned_fc2_dict(), OrderedDict()

    grads_fc3 = autograd.grad(loss, model['G'].fc3.parameters(), allow_unused=True, retain_graph=True)
    fast_weights_fc3, orderd_params_fc3 = model['G'].cloned_fc3_dict(), OrderedDict()
    for (key, val), grad in zip(model['G'].fc3.named_parameters(), grads_fc3):
        fast_weights_fc3[key] = orderd_params_fc3[key] = val - args.task_lr * grad

    grads_conv11 = autograd.grad(loss, model['G'].conv11.parameters(), allow_unused=True, retain_graph=True)
    fast_weights_conv11, orderd_params_conv11 = model['G'].cloned_conv11_dict(), OrderedDict()
    for (key, val), grad in zip(model['G'].conv11.named_parameters(), grads_conv11):
        fast_weights_conv11[key] = orderd_params_conv11[key] = val - args.task_lr * grad

    grads_conv12 = autograd.grad(loss, model['G'].conv12.parameters(), allow_unused=True, retain_graph=True)
    fast_weights_conv12, orderd_params_conv12 = model['G'].cloned_conv12_dict(), OrderedDict()
    for (key, val), grad in zip(model['G'].conv12.named_parameters(), grads_conv12):
        fast_weights_conv12[key] = orderd_params_conv12[key] = val - args.task_lr * grad

    grads_conv13 = autograd.grad(loss, model['G'].conv13.parameters(), allow_unused=True, retain_graph=True)
    fast_weights_conv13, orderd_params_conv13 = model['G'].cloned_conv13_dict(), OrderedDict()

    fast_weights = {}
    fast_weights['fc1'] = fast_weights_fc1
    fast_weights['fc2'] = fast_weights_fc2
    fast_weights['fc3'] = fast_weights_fc3
    fast_weights['conv11'] = fast_weights_conv11
    fast_weights['conv12'] = fast_weights_conv12
    fast_weights['conv13'] = fast_weights_conv13

    '''steps remaining'''
    for k in range(args.test_iter - 1):
        ancher_feature, pos_feature, neg_feature = model['G'](ancher, pos, neg, param=fast_weights)
        class_feature = model['G'].forward2(class_names_dict, param=fast_weights)
        class_loss = cs(class_feature)

        support_feature = model['G'].forward2(support, param=fast_weights)
        support_feature = support_feature.reshape(args.way, args.shot, -1)
        support_feature = support_feature.mean(1)
        support_loss = cs.forward(support_feature)

        loss_weight = torch.cat(  
        (torch.ones([(args.way-1) * support['text'].shape[0]]), args.train_loss_weight * torch.ones([20])), 0)
        if args.cuda != -1:
            loss_weight = loss_weight.cuda(args.cuda)

        # 计算损失值
        loss = criterion(ancher_feature, pos_feature, neg_feature, loss_weight=loss_weight)
        loss = loss + args.class_loss_weight * class_loss + args.proto_loss_weight * support_loss

        zero_grad(orderd_params_fc1.values())
        zero_grad(orderd_params_fc2.values())
        zero_grad(orderd_params_fc3.values())
        zero_grad(orderd_params_conv11.values())
        zero_grad(orderd_params_conv12.values())
        zero_grad(orderd_params_conv13.values())
        grads_fc3 = torch.autograd.grad(loss, orderd_params_fc3.values(), allow_unused=True, retain_graph=True)
        grads_conv11 = torch.autograd.grad(loss, orderd_params_conv11.values(), allow_unused=True, retain_graph=True)
        grads_conv12 = torch.autograd.grad(loss, orderd_params_conv12.values(), allow_unused=True, retain_graph=True)

        for (key, val), grad in zip(orderd_params_fc1.items(), grads_fc1):
            if grad is not None:
                fast_weights['fc1'][key] = orderd_params_fc1[key] = val - args.task_lr * grad

        for (key, val), grad in zip(orderd_params_fc2.items(), grads_fc2):
            if grad is not None:
                fast_weights['fc2'][key] = orderd_params_fc2[key] = val - args.task_lr * grad

        for (key, val), grad in zip(orderd_params_fc3.items(), grads_fc3):
            if grad is not None:
                fast_weights['fc3'][key] = orderd_params_fc3[key] = val - args.task_lr * grad

        for (key, val), grad in zip(orderd_params_conv11.items(), grads_conv11):
            if grad is not None:
                fast_weights['conv11'][key] = orderd_params_conv11[key] = val - args.task_lr * grad

        for (key, val), grad in zip(orderd_params_conv12.items(), grads_conv12):
            if grad is not None:
                fast_weights['conv12'][key] = orderd_params_conv12[key] = val - args.task_lr * grad

        for (key, val), grad in zip(orderd_params_conv13.items(), grads_conv13):
            if grad is not None:
                fast_weights['conv13'][key] = orderd_params_conv13[key] = val - args.task_lr * grad

    """计算Q上的损失"""
    CN = model['G'].forward_once_with_param(class_names_dict, fast_weights)
    XQ = model['G'].forward_once_with_param(query, fast_weights)

    logits_q = pos_dist(XQ, CN)
    # 平均化
    logits_q = dis_to_level(logits_q)

    _, pred = torch.max(logits_q, 1)
    acc_q = model['G'].accuracy(pred, YQ)

    return acc_q

def test(test_data, class_names, optG, model, criterion, args, test_epoch, verbose=True):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy
        and its std.
        @:param class_names、test_data、
        @:param criterion 对比损失函数
        @:param args 超参数们
        @:param test_epoch:
        @:param verbose False
    '''
    # model['G'].train()
    cs = SupConLoss(temperature=args.temperature, args=args)

    acc = []

    for ep in range(test_epoch):

        sampled_classes, source_classes = task_sampler(test_data, args)

        train_gen = SerialSampler(test_data, args, sampled_classes, source_classes, 1)

        sampled_tasks = train_gen.get_epoch()

        for task in sampled_tasks:
            if task is None:
                break
            q_acc = test_one(task, class_names, model, optG, criterion, args, grad={}, cs=cs)
            acc.append(q_acc.cpu().item())

    acc = np.array(acc)

    return np.mean(acc), np.std(acc)
