import torch
import torch.nn as nn
from torch.nn import Linear
from torch import Tensor
from torch_geometric.nn import global_add_pool, TransformerConv
from torch_geometric.nn.conv import MessagePassing


class HILNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):  # 48 256
        super().__init__()
        self.lin_node = nn.Sequential(Linear(in_channels, hidden_channels), nn.SiLU())

        self.conv_1 = InteractionLayer(hidden_channels, hidden_channels)
        self.conv_2 = InteractionLayer(hidden_channels, hidden_channels)
        self.conv_3 = InteractionLayer(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, pos, edge_attr = data.x, data.edge_index, data.pos, data.edge_attr
        x = self.lin_node(x)

        x = self.conv_1(x, edge_index, pos)
        x = self.conv_2(x, edge_index, pos)
        x = self.conv_3(x, edge_index, pos)
        x = global_add_pool(x, data.batch)
        return x


class InteractionLayer(MessagePassing):
    def __init__(self, in_channels: int,
                 out_channels: int):
        super(InteractionLayer, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mlp_node_cov = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))

        self.mlp_coord_ncov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())

    def forward(self, x, edge_index_inter, pos=None,
                size=None):
        row_ncov, col_ncov = edge_index_inter
        coord_diff_ncov = pos[row_ncov] - pos[col_ncov]
        radial_ncov = self.mlp_coord_ncov(
            _rbf(torch.norm(coord_diff_ncov, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device))
        out_node = self.propagate(edge_index=edge_index_inter, x=x, radial=radial_ncov, size=size)
        out_node = self.mlp_node_cov(x + out_node)
        return out_node

    def message(self, x_j: Tensor, x_i: Tensor, radial, index: Tensor):
        x = x_j * radial
        return x


def _rbf(D, D_min=0., D_max=6., D_count=9, device='cpu'):
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF
