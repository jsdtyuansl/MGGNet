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


def swish(x):
    return x * torch.sigmoid(x)


class emb(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super(emb, self).__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)

    def forward(self, dist, angle, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        return dist_emb, angle_emb


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


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

    def forward(self, x: Tensor, edge_index, rbf: Tensor, sbf: Tensor, idx_kj: Tensor,
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

        # Aggregate interactions and up-project embeddings:
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce='sum')
        x_kj = self.act(self.lin_up(x_kj))  #

        h = x_ji + x_kj
        h = self.act(self.lin(h)) + x
        h = self.lin_out(h)
        return h


class OutputPPBlock(torch.nn.Module):
    def __init__(
            self,
            num_radial: int,
            hidden_channels: int,
            out_emb_channels: int,
            out_channels: int,
            num_layers: int
    ):

        super().__init__()

        self.act = nn.SiLU()
        self.lin_rbf = Linear(hidden_channels, hidden_channels, bias=False)

        self.lin_up = Linear(hidden_channels, out_emb_channels, bias=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(out_emb_channels, out_emb_channels))
        self.lin = Linear(out_emb_channels, out_channels, bias=False)

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor,
                num_nodes: Optional[int] = None) -> Tensor:
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce='sum')
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class Q_DimeNetPP(torch.nn.Module):
    def __init__(
            self, in_channels=48, cutoff=5.0, num_layers=4,
            hidden_channels=128, out_channels=1, int_emb_size=128, basis_emb_size=8, out_emb_channels=256,
            num_spherical=7, num_radial=6, envelope_exponent=5,
            num_before_skip=1, num_after_skip=2, num_output_layers=3,
            act=swish, output_init='GlorotOrthogonal', num_blocks=3, num=-1, n=30, sort_by_angle=True):
        super(Q_DimeNetPP, self).__init__()

        self.cutoff = cutoff
        self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)
        self.lin_node = Linear(48, hidden_channels, bias=False)
        self.lin_edge = Linear(12, hidden_channels, bias=False)
        self.lin_rbf = Linear(6, hidden_channels, bias=False)
        self.lin_sbf = Linear(42, hidden_channels, bias=False)
        self.lin_cat = Linear(hidden_channels * 3, hidden_channels, bias=False)

        self.lin_out = nn.Sequential(Linear(hidden_channels, out_channels, bias=True),
                                     nn.Dropout(0.1),
                                     nn.LeakyReLU(),
                                     nn.BatchNorm1d(hidden_channels)
                                     )
        self.act = nn.SiLU()

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

        self.output_blocks = torch.nn.ModuleList([
            OutputPPBlock(
                num_radial,
                hidden_channels,
                out_emb_channels,
                out_channels,
                num_output_layers
            ) for _ in range(num_blocks + 1)
        ])

    def forward(self, data):
        batch = data.batch
        x = data.x
        pos = data.pos
        num_nodes = x.size(0)

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        edge_attr = self.lin_edge(edge_attr)

        dist, angle, i, j, idx_kj, idx_ji = xyz_to_dat(pos, edge_index, num_nodes,
                                                       use_torsion=False)  # idx_kj 对应50个边的索引，idx_ji对应一跳边索引
        rbf, sbf = self.emb(dist, angle, idx_kj)  # dist_emb, angle_emb  50*6   66*42

        x = self.lin_node(x)
        x = x[i]  # 544*128

        for interaction in self.interaction_blocks:
            x = interaction(x, edge_index, rbf, sbf, idx_kj, idx_ji)

        x = scatter_sum(x, index=i, dim=0, out=torch.zeros(batch.size(0), x.size(1), device=x.device))
        out = global_add_pool(x, batch)
        return out
