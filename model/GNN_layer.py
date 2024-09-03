import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


# GCNConv 简单实现
class GCNConv_s(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):

        super(GCNConv_s, self).__init__()
        self.input_dim = input_dim  # 输入维度
        self.output_dim = output_dim  # 输出维度
        self.use_bias = use_bias  # 偏置
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))  # 初始权重
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))  # 偏置
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # 重新设置参数
        # 进行凯明初始化
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            # 偏置先全给0
            nn.init.zeros_(self.bias)

    def forward(self, adjacency, input_feature, l=1):
        '''

        :param adjacency: 邻接矩阵
        :param input_feature: 输入特征
        :param l: lambda 影响自环权重值
        :return:
        '''
        # 公式: (D^-0.5) A' (D^-0.5) X W
        size = adjacency.shape[0]
        # X W
        support = torch.mm(input_feature, self.weight)

        # A' = A + \lambda I
        A = adjacency + l * torch.eye(size)

        # D: degree
        SUM = A.sum(dim=1)
        D = torch.diag_embed(SUM)
        # D'=D^(-0.5)
        D = D.__pow__(-0.5)
        # 让inf值变成0
        D[D == float("inf")] = 0

        # (D^-0.5) A' (D^-0.5)
        adjacency = torch.sparse.mm(D, adjacency)
        adjacency = torch.sparse.mm(adjacency, D)

        # (D^-0.5) A' (D^-0.5) X W
        output = torch.sparse.mm(adjacency, support)

        if self.use_bias:
            # 使用偏置
            output += self.bias
        return output

    def __repr__(self):
        # 打印的时候内存信息属性
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout  # drop prob = 0.6
        self.in_features = in_features  #
        self.out_features = out_features  #
        self.alpha = alpha  # LeakyReLU with negative input slope, alpha = 0.2
        self.concat = concat  # conacat = True for all layers except the output layer.

        # Xavier Initialization of Weights
        # Alternatively use weights_init to apply weights of choice
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):

        # 将边索引转换为稠密邻接矩阵，并去除多余的维度
        # adj = to_dense_adj(adj).squeeze(0)


        # Linear Transformation
        h = torch.mm(input, self.W)  # matrix multiplication
        N = h.size()[0]
        # print(N)

        # Attention Mechanism
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # Masked Attention
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class MH_GAT(nn.Module):
    def __init__(self, input_feature_size, output_size, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(MH_GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GATLayer(input_feature_size, output_size, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATLayer(output_size * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        print(x.size())
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)













