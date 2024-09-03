import torch
from torch import nn
from torch.nn import Linear, Embedding
from torch_geometric.nn.inits import glorot_orthogonal
from torch_scatter import scatter, scatter_sum
from math import sqrt
from torch import Tensor
# from .compute import xyz_to_dat
from .compute import xyz_to_dat
from .features import dist_emb, angle_emb
from torch_geometric.nn import MessagePassing
from typing import Callable, Optional
from model.attention import *
from torch_geometric.nn import global_add_pool


class emb(nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super(emb, self).__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)

    def forward(self, dist, angle, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        return dist_emb, angle_emb


class ResidualLayer(nn.Module):
    def __init__(self, hidden_channels):
        super(ResidualLayer, self).__init__()
        self.act = nn.SiLU()
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

        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)

        self.lin_down = Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = Linear(int_emb_size, hidden_channels, bias=False)

        self.lin = Linear(hidden_channels, hidden_channels)

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels) for _ in range(num_before_skip)
        ])
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels) for _ in range(num_after_skip)
        ])

    def forward(self, x: Tensor, rbf: Tensor, sbf: Tensor, idx_kj: Tensor,
                idx_ji: Tensor, ) -> Tensor:
        # x既是mji，也是 mkj
        x_ji = self.act(self.lin_ji(x))  # Ne*128
        x_kj = self.act(self.lin_kj(x))  # Ne*128

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

        # 消息传递得到message_kj
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce='mean')
        x_kj = self.act(self.lin_up(x_kj))

        # 得到新更新消息 m_ji
        h = x_ji + x_kj

        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)
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
        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
        self.lin_up = Linear(hidden_channels, out_emb_channels, bias=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(out_emb_channels, out_emb_channels))
        self.lin = Linear(out_emb_channels, out_channels, bias=False)

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor,
                num_nodes: int) -> Tensor:
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce='sum')
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, edge_dim: int, hidden_channels: int):
        super().__init__()
        # self.act = nn.SiLU()
        self.lin_node = nn.Sequential(Linear(44, hidden_channels), nn.SiLU())
        self.lin_edge = nn.Sequential(Linear(edge_dim, hidden_channels), nn.SiLU())
        self.lin = nn.Sequential(Linear(3 * hidden_channels, hidden_channels), nn.SiLU())

    def forward(self, x: Tensor, edge_attr: Tensor, i: Tensor, j: Tensor):
        x = self.lin_node(x)
        edge_attr = self.lin_edge(edge_attr)
        return self.lin(torch.cat([x[i], x[j], edge_attr], dim=-1))


class Q_DimeNetPP(torch.nn.Module):
    def __init__(
            self, in_channels=44, cutoff=5.0, num_layers=3,
            hidden_channels=256, out_channels=256, int_emb_size=128, basis_emb_size=8, out_emb_channels=256,
            num_spherical=7, num_radial=9, envelope_exponent=5,
            num_before_skip=0, num_after_skip=0, num_output_layers=3, edge_dim=12, num_blocks=3):
        super(Q_DimeNetPP, self).__init__()

        self.lin_node = nn.Sequential(Linear(44, hidden_channels), nn.SiLU())
        self.conv_1 = InteractionLayer(hidden_channels, hidden_channels)
        self.conv_2 = InteractionLayer(hidden_channels, hidden_channels)
        self.conv_3 = InteractionLayer(hidden_channels, out_channels)

        self.cutoff = cutoff
        self.act = nn.SiLU()
        self.emb1 = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)
        self.emb2 = EmbeddingBlock(edge_dim, hidden_channels)

        self.attention = GatedAttention(256)
        self.lin_com = nn.Sequential(Linear(hidden_channels, hidden_channels), nn.SiLU())
        self.lin_out = nn.Sequential(Linear(hidden_channels, hidden_channels), nn.SiLU())

        self.block_outs = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_channels, out_channels),
                           nn.LeakyReLU()) for _ in range(num_blocks)])

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
        batch, x, pos, edge_index, edge_attr = data.batch, data.x, data.pos, data.edge_index, data.edge_attr
        num_nodes = x.size(0)

        x1 = self.lin_node(x)
        x1 = self.conv_1(x1, edge_index, pos)
        x1 = self.conv_2(x1, edge_index, pos)
        x1 = self.conv_3(x1, edge_index, pos)

        # 从原子k到j的所有索引 从原子j到i的所有索引
        dist, angle, i, j, idx_kj, idx_ji = xyz_to_dat(pos, edge_index, num_nodes,
                                                       use_torsion=False)  # idx_kj 对应边的索引，idx_ji对应一跳边索引

        rbf, sbf = self.emb1(dist, angle, idx_kj)  # dist_emb, angle_emb

        e = self.emb2(x, edge_attr, i, j)  # Ne*128
        x2 = self.output_blocks[0](e, rbf, i, num_nodes=num_nodes)

        # x是每个边初始化完的消息
        for interaction_block, output_block, block_out in zip(self.interaction_blocks, self.output_blocks[1:],
                                                              self.block_outs):
            e = interaction_block(e, rbf, sbf, idx_kj, idx_ji)
            x2 = x2 + output_block(e, rbf, i, num_nodes=num_nodes)
            x2 = block_out(x2)

        p, _ = self.attention(torch.stack([x1, x2], dim=1))
        # p = self.lin_com(torch.cat((x1 + x2), dim=1))

        out = scatter(x1, batch, dim=0, reduce='sum')
        return out


class InteractionLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(InteractionLayer, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp_base = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())
        self.mlp_node_cov = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(self.out_channels))

    def forward(self, x, edge_index, pos, size=None):
        src, tar = edge_index
        coord_diff = pos[src] - pos[tar]
        base = self.mlp_base(
            _rbf(torch.norm(coord_diff, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device))
        out_node = self.propagate(edge_index=edge_index, x=x, base=base)
        out_node = self.mlp_node_cov(x + out_node)
        return out_node

    def message(self, x_j, base):
        x = x_j * base
        return x


def _rbf(D, D_min=0., D_max=6., D_count=9, device='cuda'):
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF
