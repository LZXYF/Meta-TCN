import argparse

from termcolor import colored
import torch
import random
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Few Shot Text Classification with Stable-PROTO.")

    parser.add_argument("--data_path", type=str,
                        default="../data/huffpost.json",
                        help="path to dataset")
    parser.add_argument("--dataset", type=str, default="huffpost",
                        help="name of the dataset. "
                             "Options: [20newsgroup, amazon, huffpost, "
                             "reuters, rcv1, fewrel]")
    parser.add_argument("--n_train_class", type=int, default=20,
                        help="number of meta-train classes")
    parser.add_argument("--n_val_class", type=int, default=5,
                        help="number of meta-val classes")
    parser.add_argument("--n_test_class", type=int, default=16,
                        help="number of meta-test classes")

    parser.add_argument("--n_workers", type=int, default=1,
                        help="Num. of cores used for loading data. Set this "
                             "to zero if you want to use all the cpus.")

    parser.add_argument("--way", type=int, default=5,
                        help="#classes for each task")
    parser.add_argument("--shot", type=int, default=1,
                        help="#support examples for each class for each task")
    parser.add_argument("--query", type=int, default=25,
                        help="#query examples for each class for each task")

    parser.add_argument("--train_epochs", type=int, default=6000,
                        help="max num of training epochs")
    parser.add_argument("--train_episodes", type=int, default=1,
                        help="#tasks sampled during each training epoch")
    parser.add_argument("--val_epochs", type=int, default=200,
                        help="#asks sampled during each validation epoch")
    parser.add_argument("--test_epochs", type=int, default=1000,
                        help="#tasks sampled during each testing epoch")

    parser.add_argument("--wv_path", type=str,
                        default='../pretrain_wordvec',
                        help="path to word vector cache")
    parser.add_argument("--word_vector", type=str, default='../pretrain_wordvec/wiki.en.vec',
                        help=("Name of pretrained word embeddings."))
    parser.add_argument("--finetune_ebd", action="store_true", default=False,
                        help=("Finetune embedding during meta-training"))

    parser.add_argument("--seed", type=int, default=2110, help="seed")
    parser.add_argument("--dropout", type=float, default=0.2, help="drop rate")
    parser.add_argument("--patience", type=int, default=8, help="patience")
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="gradient clipping")
    parser.add_argument("--cuda", type=int, default=0,
                        help="cuda device, -1 for cpu")
    parser.add_argument("--mode", type=str, default="train",
                        help=("Running mode."
                              "Options: [train, test]" 
                              "[Default: test]"))
    parser.add_argument("--save", action="store_true", default=False,
                        help="train the mode l")
    parser.add_argument("--notqdm", action="store_true", default=True,
                        help="disable tqdm")
    parser.add_argument("--result_path", type=str, default="")

    parser.add_argument("--pretrain", type=str, default=None, help="path to the pretraiend weights for MLADA")
    parser.add_argument("--train_iter", type=int, default=20, help="Number of iterations of training(in)")
    parser.add_argument("--test_iter", type=int, default=20, help="Number of iterations of testing(in)")
    parser.add_argument("--meta_lr", type=float, default=1e-5, help="learning rate of meta(out)")
    parser.add_argument("--task_lr", type=float, default=0.04, help="learning rate of task(in)")
    parser.add_argument("--task_num", type=int, default=4, help="Number of tasks")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")

    # TextCNN
    parser.add_argument("--kernel_num", type=int, default=22, help=" kernel number: output size of one kernel")
    parser.add_argument("--kernel_size", nargs='+', type=int, default=[2, 2, 4], help="kernel size list")

    parser.add_argument("--train_loss_weight", type=float, default=5.0,
                        help="the loss_weight of classname part, default the lossweight of support is 1")
    parser.add_argument("--STS", action="store_true", default=False, help="Strategy for task sampling")
    parser.add_argument("--SI", type=int, default=5, help="Sampling interval")
    parser.add_argument("--temperature", type=float, default=5.0, help="Constrast temperature")

    parser.add_argument("--savepath", type=str, default="model01", help="Save path")
    parser.add_argument("--margin", type=int, default=5.0, help="")
    parser.add_argument("--class_loss_weight", type=float, default=0.8, help="Class loss weight")
    parser.add_argument("--proto_loss_weight", type=float, default=0.1, help="Proto loss weight")
    parser.add_argument("--ebd_len", type=int, default=0, help="self.ebd_begin_len")

    args = parser.parse_args()

    args = print_args(args)

    return args


def print_args(args):
    """
        Print arguments (only show the relevant arguments)
    """

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    print(colored("""
                         __                   ______ ____    __  __     
         /'\_/`\        /\ \__               /\__  _/\  _`\ /\ \/\ \    
        /\      \     __\ \ ,_\    __        \/_/\ \\\\ \ \/\_\ \ `\\\\ \   
        \ \ \__\ \  /'__`\ \ \/  /'__`\   ______\ \ \\\\ \ \/_/\ \ , ` \  
         \ \ \_/\ \/\  __/\ \ \_/\ \L\.\_/\______\ \ \\\\ \ \L\ \ \ \`\ \ 
          \ \_\\\\ \_\ \____\\\\ \__\ \__/.\_\/______/\ \_\\\\ \____/\ \_\ \_\\
           \/_/ \/_/\/____/ \/__/\/__/\/_/         \/_/ \/___/  \/_/\/_/


                """, 'red'))

    args = print_arg(args)

    return args

def set_seed(seed):
    """
        Setting random seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model_state_dict(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    keys = []
    for k, v in pretrained_dict.items():
        keys.append(k)

    i = 0
    print("_____________pretrain_parameters______________________________")

    for k, v in model_dict.items():
        if v.size() == pretrained_dict[keys[i]].size():
            model_dict[k] = pretrained_dict[keys[i]]
            print(model_dict[k])
            i = i + 1
        # print(model_dict[k])

    print("__________________________________________________________")
    model.load_state_dict(model_dict)
    return model


def print_arg(args):
    """
        Setting random seeds
    """
    return args

def neg_dist(instances, class_proto):  # ins:[N*K, 256], cla:[N, 256]
    return -torch.pow(torch.pow(class_proto.unsqueeze(0) - instances.unsqueeze(1), 2).sum(-1), 0.5)

def pos_dist(instances, class_proto):  # ins:[N*K, 256], cla:[N, 256]
    return torch.pow(torch.pow(class_proto.unsqueeze(0) - instances.unsqueeze(1), 2).sum(-1), 0.5)

def reidx_y(args, YS, YQ):
    '''
        Map the labels into 0,..., way
        @param YS: batch_size
        @param YQ: batch_size
        @return YS_new: batch_size
        @return YQ_new: batch_size
    '''
    unique1, inv_S = torch.unique(YS, sorted=True, return_inverse=True)
    unique2, inv_Q = torch.unique(YQ, sorted=True, return_inverse=True)

    if len(unique1) != len(unique2):
        raise ValueError(
            'Support set classes are different from the query set')

    if len(unique1) != args.way:
        print("unique1", unique1)
        print("inv_S", inv_S)
        raise ValueError(
            'Support set classes are different from the number of ways')

    if int(torch.sum(unique1 - unique2).item()) != 0:
        raise ValueError(
            'Support set classes are different from the query set classes')

    Y_new = torch.arange(start=0, end=args.way, dtype=unique1.dtype,
                         device=unique1.device)

    return Y_new[inv_S], Y_new[inv_Q]
