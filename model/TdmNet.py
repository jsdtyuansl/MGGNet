import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv, CGConv, HeteroConv, Linear, SAGEConv, GATConv, global_max_pool, MLP, \
    AttentiveFP, global_mean_pool, BatchNorm, HANConv, HGTConv, GCN, HEATConv, GlobalAttention
from model.ComMNet import *
from torch_geometric.nn import global_add_pool
from model.HomoNet import HomoNet
from model.HeteroNet import HeteroNet
from model.DimeNetPP import Q_DimeNetPP
from .xKAN.FastKAN import FastKANLayer


class Tdm_Net(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        # self.ligandnet = HomoNet(in_channels=44, hidden_channels=256, out_channels=256)
        self.ligandnet = Q_DimeNetPP(in_channels=44, hidden_channels=hidden_channels, out_channels=256)

        # self.proteinnet = AttentiveFP(in_channels=44, hidden_channels=256, out_channels=256, edge_dim=12,
        #                               num_timesteps=3,
        #                               num_layers=3)
        self.proteinnet = Q_DimeNetPP(in_channels=44, hidden_channels=hidden_channels, out_channels=256)
        # self.proteinnet = HomoNet(in_channels=44, hidden_channels=256, out_channels=256)

        self.complexnet = HeteroNet(edge_dim=1, hidden_channels=hidden_channels, out_channels=256)
        self.fusion_lin = nn.Linear(3 * 256, hidden_channels)
        self.out = RegressionLayer(in_channels=256, hidden_list=[256, 512, 128], dropout=0.1, out_dim=1)
        self.fusion = 'Sum'

    def forward(self, data):
        g_l, g_p, g_com = data

        xl = self.ligandnet(g_l)
        xp = self.proteinnet(g_p)  # 取蛋白质图, torch.Size([5, 16])
        # xp = self.proteinnet(x=g_p.x, edge_index=g_p.edge_index, edge_attr=g_p.edge_attr, batch=g_p.batch)

        xc = self.complexnet(g_com.x_dict, g_com.edge_index_dict, g_com.edge_attr_dict, g_com.batch_dict)

        if self.fusion == 'Sum':
            x = xl + xp + xc
        elif self.fusion == 'Dot':
            x = xl * xp * xc
        elif self.fusion == 'Concat':
            x = torch.cat((xl, xp, xc), dim=1)
            x = self.fusion_lin(x)
        elif self.fusion == 'Average':
            x = (xl + xp + xc) / 3

        y_hat = self.out(x)
        return y_hat.view(-1)


class RegressionLayer(nn.Module):
    def __init__(self, in_channels, hidden_list, dropout, out_dim):
        super().__init__()
        self.predict = nn.ModuleList()
        for hidden_channels in hidden_list:
            self.predict.append(nn.Linear(in_channels, hidden_channels))
            self.predict.append(nn.ReLU())
            self.predict.append(nn.Dropout(dropout))
            in_channels = hidden_channels
        self.predict.append(nn.Linear(in_channels, out_dim))

    def forward(self, x):
        for fc in self.predict:
            x = fc(x)
        return x
