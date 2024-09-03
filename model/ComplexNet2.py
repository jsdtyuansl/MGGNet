from .features import dist_emb, angle_emb, torsion_emb
import torch
from torch import nn
from torch.nn import Linear, Embedding
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn import radius_graph
from torch_scatter import scatter, scatter_sum
from math import sqrt
from torch import Tensor
from .compute import xyz_to_dat
from .features import dist_emb, angle_emb
from torch_geometric.nn import DimeNet, DimeNetPlusPlus
from typing import Callable, Dict, Optional, Tuple, Union
from torch_geometric.nn import global_add_pool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def swish(x):
    m = torch.nn.LeakyReLU(0.1)
    return m(x)


class emb(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super(emb, self).__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.torsion_emb = torsion_emb(num_spherical, num_radial, cutoff, envelope_exponent)

    def forward(self, dist, angle, torsion, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        torsion_emb = self.torsion_emb(dist, angle, torsion, idx_kj)
        return dist_emb, angle_emb, torsion_emb


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


# class update_e(torch.nn.Module):
#     def __init__(self, hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion,
#                  num_spherical, num_radial,
#                  num_before_skip, num_after_skip, act=swish):
#         super(update_e, self).__init__()
#         self.act = act
#         self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
#         self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
#         self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size_angle, bias=False)
#         self.lin_sbf2 = nn.Linear(basis_emb_size_angle, int_emb_size, bias=False)
#         self.lin_t1 = nn.Linear(num_spherical * num_spherical * num_radial, basis_emb_size_torsion, bias=False)
#         self.lin_t2 = nn.Linear(basis_emb_size_torsion, int_emb_size, bias=False)
#         self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)
#
#         self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
#         self.lin_ji = nn.Linear(hidden_channels, hidden_channels)
#
#         self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)  # 128 * 64
#         self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)
#
#         self.layers_before_skip = torch.nn.ModuleList([
#             ResidualLayer(hidden_channels, act)
#             for _ in range(num_before_skip)
#         ])
#         self.lin = nn.Linear(hidden_channels, hidden_channels)
#         self.layers_after_skip = torch.nn.ModuleList([
#             ResidualLayer(hidden_channels, act)
#             for _ in range(num_after_skip)
#         ])
#
#     def forward(self, x, emb, idx_kj, idx_ji):
#         rbf0, sbf, t = emb
#         x1, _ = x  # 初始化边的时候的第一个边e1
#
#         x_ji = self.act(self.lin_ji(x1))  # 1960 *128？
#         x_kj = self.act(self.lin_kj(x1))  # 1960 * 128
#
#         rbf = self.lin_rbf1(rbf0)  # 1960 * 8
#         rbf = self.lin_rbf2(rbf)  # 1960 * 128
#         x_kj = x_kj * rbf  # 1960 * 128
#
#         x_kj = self.act(self.lin_down(x_kj))  # 1960 * 64
#
#         sbf = self.lin_sbf1(sbf)  # 36014*42
#         sbf = self.lin_sbf2(sbf)  # 36014*64
#         x_kj = x_kj[idx_kj] * sbf  # 36014*64
#
#         t = self.lin_t1(t)
#         t = self.lin_t2(t)
#         x_kj = x_kj * t  # 36014*64
#
#         x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))  # 1960
#         x_kj = self.act(self.lin_up(x_kj))  # 1960 * 128
#
#         e1 = x_ji + x_kj
#         for layer in self.layers_before_skip:
#             e1 = layer(e1)
#         e1 = self.act(self.lin(e1)) + x1
#         for layer in self.layers_after_skip:
#             e1 = layer(e1)
#         e2 = self.lin_rbf(rbf0) * e1
#
#         return e1, e2


class InteractionPPBlock(torch.nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            int_emb_size: int,
            basis_emb_size: int,
            num_spherical: int,
            num_radial: int,
            num_before_skip: int,
            num_after_skip: int
    ):
        super().__init__()
        self.act = nn.SiLU()

        # Transformation of Bessel and spherical basis representations:
        self.lin_rbf1 = Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = Linear(basis_emb_size, hidden_channels, bias=False)

        self.lin_sbf1 = Linear(num_spherical * num_radial, basis_emb_size,
                               bias=False)
        self.lin_sbf2 = Linear(basis_emb_size, int_emb_size, bias=False)

        self.lin_t1 = nn.Linear(num_spherical * num_spherical * num_radial, basis_emb_size, bias=False)
        self.lin_t2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)

        # Hidden transformation of input message:
        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)

        # Embedding projections for interaction triplets:
        self.lin_down = Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = Linear(int_emb_size, hidden_channels, bias=False)

        self.lin = Linear(hidden_channels, hidden_channels)

        self.lin_out = nn.Sequential(Linear(hidden_channels, hidden_channels),
                                     nn.Dropout(0.1),
                                     nn.SiLU(),
                                     nn.BatchNorm1d(hidden_channels))

    def forward(self, x: Tensor, edge_index, rbf: Tensor, sbf: Tensor, t: Tensor, idx_kj: Tensor,
                idx_ji: Tensor) -> Tensor:
        # Initial transformation:
        x_ji = self.act(self.lin_ji(x))  # 50*128
        x_kj = self.act(self.lin_kj(x))  # 50*128

        # Transformation via Bessel basis:
        rbf = self.lin_rbf1(rbf)  # 50*128
        rbf = self.lin_rbf2(rbf)  # 50*128
        x_kj = x_kj * rbf  # 50*128

        # Down project embedding and generating triple-interactions:
        x_kj = self.act(self.lin_down(x_kj))  # 50*128

        # Transform via 2D spherical basis:
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf  # 66*128

        t = self.lin_t1(t)
        t = self.lin_t2(t)
        x_kj = x_kj * t  # 36014*64

        # Aggregate interactions and up-project embeddings:
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce='sum')
        x_kj = self.act(self.lin_up(x_kj))  #

        h = x_ji + x_kj
        h = self.act(self.lin(h)) + x
        h = self.lin_out(h)
        return h



class TdComNet(torch.nn.Module):
    def __init__(
            self, in_channels=48, cutoff=5.0, num_layers=3,
            hidden_channels=128, out_channels=1, int_emb_size=64,
            basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
            num_spherical=7, num_radial=6, envelope_exponent=5,
            num_before_skip=1, num_after_skip=1, num_output_layers=1,
            act=swish, num_blocks=3, basis_emb_size=8, use_node_features=True):
        super(TdComNet, self).__init__()

        self.cutoff = cutoff
        self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)
        self.lin_node = Linear(48, hidden_channels, bias=False)

        self.interaction_blocks = torch.nn.ModuleList([
            InteractionPPBlock(
                hidden_channels,
                int_emb_size,
                basis_emb_size,
                num_spherical,
                num_radial,
                num_before_skip,
                num_after_skip,
            ) for _ in range(num_blocks)
        ])

    def forward(self, batch_data):
        x, pos, batch = batch_data.x, batch_data.pos, batch_data.batch

        edge_index = batch_data.edge_index
        num_nodes = x.size(0)

        dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(pos, edge_index, num_nodes,
                                                                use_torsion=True)  # 1960 36014 36014

        rbf, sbf, tbf = self.emb(dist, angle, torsion, idx_kj)  # dist_emb, angle_emb  50*6   66*42

        x = self.lin_node(x)
        x = x[i]  # 544*128

        for interaction in self.interaction_blocks:
            x = interaction(x, edge_index, rbf, sbf, tbf, idx_kj, idx_ji)

        x = scatter_sum(x, index=i, dim=0, out=torch.zeros(batch.size(0), x.size(1), device=x.device))
        out = global_add_pool(x, batch)
        return out
