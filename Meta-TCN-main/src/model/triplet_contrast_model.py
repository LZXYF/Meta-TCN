import torch
import torch.nn as nn

import torch.nn.functional as F


# 嵌入网络
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


# Triplet Encoder 网络
class ModelG(nn.Module):

    def __init__(self, ebd, args):
        super(ModelG, self).__init__()

        self.args = args

        # fastText
        self.ebd = ebd
        
        self.ebd_begin_len = args.ebd_len

        self.ebd_dim = self.ebd.embedding_dim
        self.hidden_size = 128

        # Text CNN
        ci = 1  # input chanel size
        kernel_num = args.kernel_num  # output chanel size
        kernel_size = args.kernel_size
        dropout = args.dropout
        self.conv11 = nn.Conv2d(ci, kernel_num, (kernel_size[0], self.ebd_dim)).cuda(args.cuda)
        self.conv12 = nn.Conv2d(ci, kernel_num, (kernel_size[1], self.ebd_dim))
        self.conv13 = nn.Conv2d(ci, kernel_num, (kernel_size[2], self.ebd_dim))

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_size) * kernel_num, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(2 * kernel_num, 64)
        self.cost = nn.CrossEntropyLoss()

    def forward_once(self, data, tmp_len=0):
        # data ：{'text': , ‘text_len’:, 'label':}
        ebd = self.ebd(data)  # [b, text_len, 300]

        if tmp_len > 0:
            ebd = ebd[:, :tmp_len, :]

        ebd = ebd.unsqueeze(1)  # [b, 1, text_len, 300]
        x1 = self.conv11(ebd.cuda(self.args.cuda))  # [b, kernel_num, H_out, 1]
        x1 = F.relu(x1.squeeze(3))  # [b, kernel_num, H_out]
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)  # [batch, kernel_num]

        x2 = self.conv12(ebd)  # [b, kernel_num, H_out, 1]
        x2 = F.relu(x2.squeeze(3))  # [b, kernel_num, H_out]
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)  # [batch, kernel_num]

        x = torch.cat((x1, x2), 1)  # [b, 2 * kernel_num]

        x = self.fc3(x)
        x = self.dropout(x) 

        return x


    def forward_once_with_param(self, data, param, tmp_len=None):

        ebd = self.ebd(data)  # [b, text_len, 300]

        ebd = ebd.unsqueeze(1)  # [b, 1, text_len, 300]

        w1, b1 = param['conv11']['weight'], param['conv11']['bias']
        x1 = F.conv2d(ebd, w1, b1)  # [b, kernel_num, H_out, 1]
        x1 = F.relu(x1.squeeze(3))  # [b, kernel_num, H_out]
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)  # [batch, kernel_num]

        w2, b2 = param['conv12']['weight'], param['conv12']['bias']
        x2 = F.conv2d(ebd, w2, b2)  # [b, kernel_num, H_out, 1]
        x2 = F.relu(x2.squeeze(3))  # [b, kernel_num, H_out]
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)  # [batch, kernel_num]

        x = torch.cat((x1, x2), 1)  # [b, 2 * kernel_num]

        w_fc3, b_fc3 = param['fc3']['weight'], param['fc3']['bias']
        x = F.linear(x, w_fc3, b_fc3)  # [b, 128]
        x = self.dropout(x)

        return x

    # 网络入口
    def forward(self, inputs_1, inputs_2, inputs_3, param=None):
        if param is None:
            out_1 = self.forward_once(inputs_1)
            out_2 = self.forward_once(inputs_2)
            out_3 = self.forward_once(inputs_3)
        else:
            out_1 = self.forward_once_with_param(inputs_1, param)
            out_2 = self.forward_once_with_param(inputs_2, param)
            out_3 = self.forward_once_with_param(inputs_3, param)
        return out_1, out_2, out_3

    def forward2(self, inputs_1, param=None):
        if param is None:
            out_1 = self.forward_once(inputs_1)
        else:
            out_1 = self.forward_once_with_param(inputs_1, param)
        return out_1

    def cloned_fc1_dict(self):
        return {key: val.clone() for key, val in self.fc1.state_dict().items()}

    def cloned_fc2_dict(self):
        return {key: val.clone() for key, val in self.fc2.state_dict().items()}

    def cloned_fc3_dict(self):
        return {key: val.clone() for key, val in self.fc3.state_dict().items()}

    def cloned_conv11_dict(self):
        return {key: val.clone() for key, val in self.conv11.state_dict().items()}

    def cloned_conv12_dict(self):
        return {key: val.clone() for key, val in self.conv12.state_dict().items()}

    def cloned_conv13_dict(self):
        return {key: val.clone() for key, val in self.conv13.state_dict().items()}

    def loss(self, logits, label):
        loss_ce = self.cost(logits, label)
        return loss_ce

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label).type(torch.FloatTensor))


    
