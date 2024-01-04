import itertools
import torch

import torch.nn.functional as F



def named_grad_param(model, keys):
    '''
        Return a generator that generates learnable named parameters in
        model[key] for key in keys.
    '''
    if len(keys) == 1:
        return filter(lambda p: p[1].requires_grad,
                model[keys[0]].named_parameters())
    else:
        return filter(lambda p: p[1].requires_grad,
                itertools.chain.from_iterable(
                    model[key].named_parameters() for key in keys))


def grad_param(model, keys):
    '''
        Return a generator that generates learnable parameters in
        model[key] for key in keys.
    '''
    if len(keys) == 1:
        return filter(lambda p: p.requires_grad,
                model[keys[0]].parameters())
    else:
        return filter(lambda p: p.requires_grad,
                itertools.chain.from_iterable(
                    model[key].parameters() for key in keys))


def get_norm(model):
    '''
        Compute norm of the gradients
    '''
    total_norm = 0

    for p in model.parameters():
        if p.grad is not None:
            p_norm = p.grad.data.norm()
            total_norm += p_norm.item() ** 2

    total_norm = total_norm ** 0.5

    return total_norm


def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

def neg_dist(instances, class_proto):  # ins:[N*K, 256], cla:[N, 256]
    return -torch.pow(torch.pow(class_proto.unsqueeze(0) - instances.unsqueeze(1), 2).sum(-1), 0.5)


def pos_dist(instances, class_proto):  # ins:[N*K, 256], cla:[N, 256]
    return torch.pow(torch.pow(class_proto.unsqueeze(0) - instances.unsqueeze(1), 2).sum(-1), 0.5)


def dis_to_level(dis):
    tmp_mean = torch.mean(dis, dim=-1, keepdim=True)
    result = dis / tmp_mean
    return -result

# TripletLoss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=10.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True, loss_weight=None):
        distance_positive = (anchor - positive).pow(2).sum(1)# .pow(.5)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)# .pow(.5) # .pow(.5)
        if loss_weight is None:
            losses = F.relu(distance_positive - distance_negative + self.margin)
        else:
            result = (distance_positive - distance_negative + self.margin) * loss_weight
            losses = F.relu(result)
        return losses.mean() if size_average else losses.sum()


def get_weight_of_test_support(support, query, args):
    """
    :param support: 5way5shot时 [5x5+5, 64] 包含 5个标签描述文本
    :param query: 5way5shot时 [5x5, 64]
    :param args:
    :return:
    """
    if len(support) > args.way*args.shot:
        # 只取出属于support部分的样本特征
        support = support[0:args.way*args.shot]

    # test_loss_weight = 5.8
    # 生成一个 长度为 150 的 前面125个全是 1，后面25个是 5.8 的向量
    result = torch.cat((torch.ones([args.way*args.shot*args.way]),
                         args.test_loss_weight*torch.ones([args.way*args.way])), 0)

    # 获取特征向量的维度长度 64
    tensor_shape = support.shape[-1]
    for each_way in range(args.way):
        # 五个五个的从support中取出来，当前取出前五个样本特征（即同属一个类）
        this_support = support[each_way*args.shot:each_way*args.shot+args.shot]
        # 同上，将同属一个类的查询样本特征取出
        this_query = query[each_way*args.query:each_way*args.query+args.query]
        #用来收集每个支持集样本(5) 到 所有查询集样本距离的平均值
        all_dis = torch.ones([args.shot])
        new_support = torch.ones([args.query, tensor_shape])
        for each_shot in range(args.shot):
            # 取出这五个（同一类）中的一个支持集样本
            new_support[:] = this_support[each_shot]
            new_support = new_support.cuda(args.cuda)
            # 刚取出的支持集样本 计算与 同类中的每个查询集的样本的距离
            this_dis = F.pairwise_distance(new_support, this_query, keepdim=True)
            # 平均距离
            this_dis = torch.mean(this_dis)
            all_dis[each_shot] = this_dis
        # 将这些（5）平均距离除以平均值，归一化大小，并返回其负值
        # 相当于计算 查询集 到 支持集 的距离，距离越大，其负值就越小，其 probab 就越小
        probab = dis_to_level(all_dis)
        probab = F.softmax(probab, dim = -1)
        probab = 5*probab

        # 赋值给 0-5 全部赋值 一个数，后面5-10全部赋值一个，依次。。。
        # 例如，上层函数中，配对后的 0-5 个 距离结果 代表第一个支持样本与每个标签文本进行匹配的结果。
        # 它的 0-25 才包含，同一类的支持集样本（5个）分别都与 标签文本（5）进行匹配的结果。
        # 而 这里 probab 中的第 1 个 表示 一个支持集 与 同一类中所有查询集样本平均距离，
        # 即这一个支持集到同类查询集样本的距离。
        # 所以 probab 中的一个就是 result（权重向量）中表示前五个权重值，即 第一个 样本的权重值，这个样本与每个标签文本都进行了依次匹配。
        for each_shot in range(args.shot):
            begin = each_way*(args.shot*args.way)+each_shot*args.way
            result[begin:begin+args.way] = probab[each_shot]

    if args.cuda != -1:
        result = result.cuda(args.cuda)
    return result

