import os
import time
import datetime

import torch
import numpy as np
import copy
import itertools
from train.ContrastFunction import SupConLoss

from train.utils import *
from dataset.sampler2 import SerialSampler,  task_sampler
from tqdm import tqdm
from termcolor import colored
from train.test import test
import torch.nn.functional as F
from dataset import utils
from tools.tool import neg_dist, reidx_y

from torch import autograd
from collections import OrderedDict


def del_tensor_ele(arr, index):
        arr1 = arr[0:index]
        arr2 = arr[index + 1:]
        return torch.cat((arr1, arr2), dim=0)

def pre_calculate_org(train_data, class_names, net, args):
    with torch.no_grad():
        all_classes = np.unique(train_data['label'])
        num_classes = len(all_classes)

        train_class_names = {}
        train_class_names['text'] = class_names['text'][all_classes]
        train_class_names['text_len'] = class_names['text_len'][all_classes]
        train_class_names['label'] = class_names['label'][all_classes]
        train_class_names = utils.to_tensor(train_class_names, args.cuda)
        train_class_names_ebd = net.ebd(train_class_names)  # [10, 36, 300]

        train_class_names_ebd = torch.sum(train_class_names_ebd, dim=1) / train_class_names['text_len'].view((-1, 1))  # [10, 300]离
        dist_metrix = neg_dist(train_class_names_ebd, train_class_names_ebd)  # [10, 10]

        for i, d in enumerate(dist_metrix):
            if i == 0:
                dist_metrix_nodiag = del_tensor_ele(d, i).view((1, -1))
            else:
                dist_metrix_nodiag = torch.cat((dist_metrix_nodiag, del_tensor_ele(d, i).view((1, -1))), dim=0)

        prob_metrix = F.softmax(dist_metrix_nodiag, dim=1)  # [10, 9]
        prob_metrix = prob_metrix.cpu().numpy()

        example_prob_metrix = []
        example_prob_metrix_nodiag = []
        for i, label in enumerate(all_classes):
            train_examples = {}
            train_examples['text'] = train_data['text'][train_data['label'] == label]
            train_examples['text_len'] = train_data['text_len'][train_data['label'] == label]
            train_examples['label'] = train_data['label'][train_data['label'] == label]
            train_examples = utils.to_tensor(train_examples, args.cuda)
            train_examples_ebd = net.ebd(train_examples)
            train_examples_ebd = torch.sum(train_examples_ebd, dim=1) / train_examples['text_len'].view(
                                    (-1, 1))  # [N, 300]

            example_prob_metrix_oe = -neg_dist(train_class_names_ebd[i].view((1, -1)), train_examples_ebd)
            example_prob_metrix_nodiag.append(example_prob_metrix_oe.cpu())
            example_prob_metrix_one = F.softmax(example_prob_metrix_oe, dim=1)  # [1, 1000]
            example_prob_metrix_one = example_prob_metrix_one.cpu().numpy()

            example_prob_metrix.append(example_prob_metrix_one)

        return prob_metrix, example_prob_metrix, -dist_metrix_nodiag.cpu(), example_prob_metrix_nodiag


def pre_calculate(train_data, class_names, net, args):
    with torch.no_grad():
        all_classes = np.unique(train_data['label'])
        num_classes = len(all_classes)

        train_class_names = {}
        train_class_names['text'] = class_names['text'][all_classes]
        train_class_names['text_len'] = class_names['text_len'][all_classes]
        train_class_names['label'] = class_names['label'][all_classes]
        train_class_names = utils.to_tensor(train_class_names, args.cuda)
        train_class_names_ebd = net.forward_once(train_class_names)

        dist_metrix = neg_dist(train_class_names_ebd, train_class_names_ebd)  # [10, 10]

        for i, d in enumerate(dist_metrix):
            if i == 0:
                dist_metrix_nodiag = del_tensor_ele(d, i).view((1, -1))
            else:
                dist_metrix_nodiag = torch.cat((dist_metrix_nodiag, del_tensor_ele(d, i).view((1, -1))), dim=0)

        prob_metrix = F.softmax(dist_metrix_nodiag, dim=1)  # [10, 9]
        prob_metrix = prob_metrix.cpu().numpy()

        example_prob_metrix = []
        example_prob_metrix_nodiag = []
        for i, label in enumerate(all_classes):
            train_examples = {}
            train_examples['text'] = train_data['text'][train_data['label'] == label]
            train_examples['text_len'] = train_data['text_len'][train_data['label'] == label]
            train_examples['label'] = train_data['label'][train_data['label'] == label]
            train_examples = utils.to_tensor(train_examples, args.cuda)
            train_examples_ebd = net.forward_once(train_examples)

            example_prob_metrix_oe = -neg_dist(train_class_names_ebd[i].view((1, -1)), train_examples_ebd)
            example_prob_metrix_nodiag.append(example_prob_metrix_oe.cpu())
            example_prob_metrix_one = F.softmax(example_prob_metrix_oe, dim=1)  # [1, 1000]
            example_prob_metrix_one = example_prob_metrix_one.cpu().numpy()

            example_prob_metrix.append(example_prob_metrix_one)

        return prob_metrix, example_prob_metrix, -dist_metrix_nodiag.cpu(), example_prob_metrix_nodiag

def pre_calculate2(dist_metrix_nodiag_0, example_prob_metrix_nodiag_0, train_data, class_names, net, args):

    with torch.no_grad():
        all_classes = np.unique(train_data['label'])
        num_classes = len(all_classes)

        train_class_names = {}
        train_class_names['text'] = class_names['text'][all_classes]
        train_class_names['text_len'] = class_names['text_len'][all_classes]
        train_class_names['label'] = class_names['label'][all_classes]
        train_class_names = utils.to_tensor(train_class_names, args.cuda)
        train_class_names_ebd = net.forward_once(train_class_names, tmp_len=args.ebd_len)

        dist_metrix = -neg_dist(train_class_names_ebd, train_class_names_ebd)  # [10, 10]
        dist_metrix = dist_metrix + 1e-9

        for i, d in enumerate(dist_metrix):
            if i == 0:

                dist_metrix_nodiag = del_tensor_ele(d, i).view((1, -1))
            else:
                dist_metrix_nodiag = torch.cat((dist_metrix_nodiag, del_tensor_ele(d, i).view((1, -1))), dim=0)

        dist_metrix = -dist_metrix_nodiag.cpu()
        prob_metrix = F.softmax(dist_metrix, dim=1)  # [10, 9]
        prob_metrix = prob_metrix.cpu().numpy()

        example_prob_metrix = []
        example_prob_metrix_nodiag = []
        for i, label in enumerate(all_classes):
            train_examples = {}
            train_examples['text'] = train_data['text'][train_data['label'] == label]
            train_examples['text_len'] = train_data['text_len'][train_data['label'] == label]
            train_examples['label'] = train_data['label'][train_data['label'] == label]
            train_examples = utils.to_tensor(train_examples, args.cuda)

            train_examples_ebd = net.forward_once(train_examples)

            example_prob_metrix_oe = -neg_dist(train_class_names_ebd[i].view((1, -1)), train_examples_ebd)

            example_prob_metrix_oe = example_prob_metrix_oe.cpu()
            example_prob_metrix_nodiag.append(example_prob_metrix_oe)

            example_prob_metrix_mean = example_prob_metrix_oe.mean(1)
            mean_metrix = example_prob_metrix_oe > example_prob_metrix_mean
            mean_metrix = mean_metrix.int()

            example_prob_metrix_one = example_prob_metrix_oe - example_prob_metrix_nodiag_0[i]
            example_prob_metrix_one = example_prob_metrix_one / example_prob_metrix_nodiag_0[i]


            example_prob_metrix_one = F.softmax(example_prob_metrix_one, dim=1)  # [1, 1000]
            example_prob_metrix_one = example_prob_metrix_one.cpu().numpy()
            example_prob_metrix.append(example_prob_metrix_one)

        return prob_metrix, example_prob_metrix, dist_metrix_nodiag.cpu(), example_prob_metrix_nodiag

def   getMystra(dist_metrix_nodiag_0, example_prob_metrix_nodiag_0, train_data, class_names, net, args):

    classes_sample_p, example_prob_metrix, dist_metrix_nodiag, example_prob_metrix_nodiag = \
        pre_calculate2(dist_metrix_nodiag_0, example_prob_metrix_nodiag_0, train_data, class_names, net, args)

    return dist_metrix_nodiag, example_prob_metrix_nodiag, classes_sample_p, example_prob_metrix



# 执行一次训练的函数
def train_one(task, class_names, model, optG, criterion, args, grad, cs):
    '''
        Train the model on one sampled task.
    '''
    model['G'].train()

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
        tmp_pos = torch.ones(args.way-1,
                             allsample['text'].shape[1]).cuda(args.cuda)

        indices = torch.nonzero(torch.eq(class_names_dict['label'], allsample['label'][i]))
        index = indices[0][0].item()
        tmp_pos = tmp_pos * class_names_dict['text'][index, :]

        if pos['text'] is None:
            pos['text'] = tmp_pos
        else:
            pos['text'] = torch.cat([pos['text'], tmp_pos], dim=0)

        if ancher["text"] is None:
            ancher['text'] = allsample['text'][i].repeat(args.way-1, 1).long()
        else:
            ancher['text'] = torch.cat([ancher['text'], allsample['text'][i].repeat(args.way-1, 1).long()], dim=0)

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

    support_feature = model['G'].forward2(support)
    support_feature = support_feature.reshape(args.way, args.shot, -1)
    support_feature = support_feature.mean(1)
    support_loss = cs.forward(support_feature)

    class_loss = cs(class_feature)

    loss_weight = torch.cat(# train_loss_weight = 10.0 （huffpost 5 5）

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
    for k in range(args.train_iter - 1):
        ancher_feature, pos_feature, neg_feature = model['G'](ancher, pos, neg, param=fast_weights)
        class_feature = model['G'].forward2(class_names_dict, param=fast_weights)
        class_loss = cs(class_feature)

        support_feature = model['G'].forward2(support, param=fast_weights)
        support_feature = support_feature.reshape(args.way, args.shot, -1)
        support_feature = support_feature.mean(1)
        support_loss = cs.forward(support_feature)

        loss_weight = torch.cat(  # train_loss_weight = 10.0 （huffpost 5 5）
            (torch.ones([(args.way-1) * support['text'].shape[0]]), args.train_loss_weight * torch.ones([20])), 0)
        if args.cuda != -1:
            loss_weight = loss_weight.cuda(args.cuda)

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


    """计算损失"""
    CN = model['G'].forward_once_with_param(class_names_dict, fast_weights)
    XQ = model['G'].forward_once_with_param(query, fast_weights)

    logits_q = pos_dist(XQ, CN)
    logits_q = dis_to_level(logits_q)

    q_loss = model['G'].loss(logits_q, YQ)

    _, pred = torch.max(logits_q, 1)
    acc_q = model['G'].accuracy(pred, YQ)

    return q_loss, acc_q


# 主训练函数
def train(train_data, val_data, model, class_names, criterion, args):
    '''
        Train the model
        Use val_data to do early stopping
        @:param class_names、train_data、...
        @:param criterion 对比损失函数
        @:param args 超参数们
    }
    '''
    # creating a tmp directory to save the models
    out_dir = os.path.abspath(os.path.join(
        os.path.curdir,
        args.savepath))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    best_acc = 0
    sub_cycle = 0
    best_path = None
    cs = SupConLoss(temperature=args.temperature, args=args)

    # 默认False
    if args.STS == True:
        classes_sample_p, example_prob_metrix, dist_metrix_nodiag, example_prob_metrix_nodiag = \
            pre_calculate(train_data, class_names, model['G'], args)
    else:
        classes_sample_p, example_prob_metrix = None, None

    # 创建优化器
    optG = torch.optim.Adam(grad_param(model, ['G']), lr=args.meta_lr, weight_decay=args.weight_decay)

    print("{}, Start training".format(
        datetime.datetime.now()), flush=True)

    acc = 0
    loss = 0
    for ep in range(args.train_epochs):
        ep_loss = 0
        #
        if (args.STS == True) and (ep != 0) and (ep % args.SI == 0):
            dist_metrix_nodiag, example_prob_metrix_nodiag, classes_sample_p, example_prob_metrix \
                = getMystra(dist_metrix_nodiag, example_prob_metrix_nodiag, train_data, class_names, model['G'], args)

        for _ in range(args.train_episodes):

            sampled_classes, source_classes = task_sampler(train_data, args, classes_sample_p)

            train_gen = SerialSampler(train_data, args, sampled_classes, source_classes, args.train_episodes, example_prob_metrix)

            sampled_tasks = train_gen.get_epoch()

            grad = {'clf': [], 'G': []}

            if not args.notqdm:
                sampled_tasks = tqdm(sampled_tasks, total=train_gen.num_episodes,
                                     ncols=80, leave=False, desc=colored('Training on train',
                                                                         'yellow'))

            for task in sampled_tasks:
                if task is None:
                    break
                q_loss, q_acc = train_one(task, class_names, model, optG, criterion, args, grad, cs)
                acc += q_acc
                loss = loss + q_loss
                ep_loss = ep_loss + q_loss

        ep_loss = ep_loss / args.train_episodes

        optG.zero_grad()
        ep_loss.backward()
        optG.step()

        test_count = 100
        if (ep % test_count == 0) and (ep != 0):
            acc = acc / args.train_episodes / test_count
            loss = loss / args.train_episodes / test_count
            print("{}:".format(colored('--------[TRAIN] ep', 'yellow')) + str(ep) + ", mean_loss:" + str(loss.item()) + ", mean_acc:" + str(
                acc.item()) + "-----------")

            # net = copy.deepcopy(model)
            net = model

            acc = 0
            loss = 0

            cur_acc, cur_std = test(val_data, class_names, optG, net, criterion, args, args.val_epochs, False)
            print(("[EVAL] {}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f}, "
                   ).format(
                datetime.datetime.now(),
                "ep", ep,
                colored("val  ", "cyan"),
                colored("acc:", "blue"), cur_acc, cur_std,
            ), flush=True)

            # Update the current best model if val acc is better
            if cur_acc > best_acc:
                best_acc = cur_acc
                best_path = out_dir

                # # save current model
                print("{}, Save cur best model to {}".format(
                    datetime.datetime.now(),
                    best_path))
                torch.save(model['G'].state_dict(), best_path + '.G')

                sub_cycle = 0
            else:
                sub_cycle += 1

            # Break if the val acc hasn't improved in the past patience epochs
            if sub_cycle == args.patience:
                break

    print("{}, End of training. Restore the best weights".format(
        datetime.datetime.now()),
        flush=True)

    if args.save:
        # save the current model
        out_dir = os.path.abspath(os.path.join(
            os.path.curdir,
            "saved-runs",
            str(int(time.time() * 1e7))))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        best_path = os.path.join(out_dir, 'best')

        print("{}, Save best model to {}".format(
            datetime.datetime.now(),
            best_path), flush=True)

        torch.save(model['G'].state_dict(), best_path + '.G')

        with open(best_path + '_args.txt', 'w') as f:
            for attr, value in sorted(args.__dict__.items()):
                f.write("{}={}\n".format(attr, value))

    return optG

