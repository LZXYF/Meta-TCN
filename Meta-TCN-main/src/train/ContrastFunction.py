import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def dot_similarity(x1, x2):
    return torch.matmul(x1, x2.t())


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

class SupConLoss(nn.Module):
    def __init__(self, temperature=5, contrast_mode='all', args=None):
        super(SupConLoss, self).__init__()
        self.tau = temperature
        self.contrast_mode = contrast_mode
        self.args = args


    def similarity(self, x1, x2):
        M = euclidean_dist(x1, x2)
        s = torch.exp(-M/self.tau)
        return s

    def forward(self, x):

        X = x
        batch_labels = torch.tensor([0,1,2,3,4]).cuda(self.args.cuda)

        len_ = batch_labels.size()[0]

        # computing similarities for each positive and negative pair
        s = self.similarity(X, X)

        # computing masks for contrastive loss
        if len(x) == 1:
            mask_i = torch.from_numpy(np.ones((len_, len_))).to(batch_labels.device)
        else:
            mask_i = 1. - torch.from_numpy(np.identity(len_)).to(batch_labels.device)  # sum over items in the numerator
        label_matrix = batch_labels.unsqueeze(0).repeat(len_, 1)
        mask_j = mask_i 
        pos_num = torch.sum(mask_j, 1)

        s_i = torch.clamp(torch.sum(s * mask_i, 1), min=1e-9)
        s_j = torch.clamp(s * mask_j, min=1e-9)
        log_p = torch.sum(-torch.log(s_j / s_i) * mask_j, 1) / pos_num
        loss = torch.mean(log_p)

        return loss

    def forward2(self, x):
        X = x

        batch_labels = torch.tensor([0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4]).cuda(self.args.cuda)
        # 10
        len_ = batch_labels.size()[0]

        # computing similarities for each positive and negative pair
        s = self.similarity(X, X)

        # computing masks for contrastive loss
        if len(x) == 1:
            mask_i = torch.from_numpy(np.ones((len_, len_))).to(batch_labels.device)
        else:
            mask_i = 1. - torch.from_numpy(np.identity(len_)).to(batch_labels.device)  # sum over items in the numerator

        label_matrix = batch_labels.unsqueeze(0).repeat(len_, 1)
        mask_j = (batch_labels.unsqueeze(1) - label_matrix == 0).float() * mask_i  # sum over items in the denominator
        mask_j = 1.0 - mask_j
        pos_num = torch.sum(mask_j, 1)

        s_i = torch.clamp(torch.sum(s * mask_i, 1), min=1e-9)
        s_j = torch.clamp(s * mask_j, min=1e-9)
        log_p = torch.sum(-torch.log(s_j / s_i) * mask_j, 1) / pos_num
        loss = torch.mean(log_p)

        return loss
