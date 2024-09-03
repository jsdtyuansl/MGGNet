import torch
from torch import nn
from torch.nn import Linear, Embedding
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
from math import sqrt
from .compute import *
from .utils.geometric_computing import xyz_to_dat
from .features import *

torch.set_printoptions(precision=4, sci_mode=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    def __init__(self, hidden_channels):
        super(ResidualLayer, self).__init__()
        self.act = nn.SELU()
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class init(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, use_node_features=True):
        super(init, self).__init__()
        self.act = nn.SELU()
        self.use_node_features = use_node_features

        self.node_lin = Linear(48, hidden_channels)

        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)

    def forward(self, x, emb, i, j):
        rbf, _, _ = emb  # 1960*6

        x = self.node_lin(x)
        rbf0 = self.act(self.lin_rbf_0(rbf))  # 1960*128
        e1 = self.act(
            self.lin(torch.cat([x[i], x[j], rbf0], dim=-1)))  # 1960*128  1960*128  1960*128  将两个节点和边转换后的rbf0学习得到边初始边
        e2 = self.lin_rbf_1(rbf) * e1  # 有必要吗   更新后的边乘变换后的rbf
        return e1, e2


class update_e(torch.nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion,
                 num_spherical, num_radial,
                 num_before_skip, num_after_skip):
        super(update_e, self).__init__()
        self.act = nn.SELU()
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size_angle, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size_angle, int_emb_size, bias=False)
        self.lin_t1 = nn.Linear(num_spherical * num_spherical * num_radial, basis_emb_size_torsion, bias=False)
        self.lin_t2 = nn.Linear(basis_emb_size_torsion, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)  # 128 * 64
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.lin_cat = Linear(hidden_channels + 1, hidden_channels, bias=False)

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels)
            for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels)
            for _ in range(num_after_skip)
        ])

    def forward(self, x, emb, idx_kj, idx_ji, q_r):
        rbf0, sbf, t = emb
        x1, _ = x  # 初始化边的时候的第一个边e1

        x_ji = self.act(self.lin_ji(x1))  # 1960 *128？
        x_kj = self.act(self.lin_kj(x1))  # 1960 * 128

        rbf = self.lin_rbf1(rbf0)  # 1960 * 8
        rbf = self.lin_rbf2(rbf)  # 1960 * 128
        x_kj = x_kj * rbf  # 1960 * 128

        x_kj = self.act(self.lin_down(x_kj))  # 1960 * 64

        sbf = self.lin_sbf1(sbf)  # 36014*42
        sbf = self.lin_sbf2(sbf)  # 36014*64
        x_kj = x_kj[idx_kj] * sbf  # 36014*64

        t = self.lin_t1(t)
        t = self.lin_t2(t)
        x_kj = x_kj * t  # 36014*64

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))  # 1960
        x_kj = self.act(self.lin_up(x_kj))  # 1960 * 128

        e1 = x_ji + x_kj

        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1
        return e1, e2


class update_v(torch.nn.Module):
    def __init__(self, hidden_channels, out_emb_channels, out_channels, num_output_layers):
        super(update_v, self).__init__()
        self.act = nn.SiLU()

        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

    def forward(self, e, i):
        _, e2 = e  # 再经过rbf后的边（消息）  1960 * 128   1960？
        v = scatter(e2, i, dim=0)  # 根据i的索引进行聚合
        v = self.lin_up(v)  # 115*256
        for lin in self.lins:
            v = self.act(lin(v))
        v = self.lin(v)
        return v


class update_u(torch.nn.Module):
    def __init__(self):
        super(update_u, self).__init__()

    def forward(self, u, v, batch):
        u += scatter(v, batch, dim=0)
        return u


class TdComNet(torch.nn.Module):
    def __init__(
            self, in_channels=48, cutoff=5.0, num_layers=3,
            hidden_channels=128, out_channels=1, int_emb_size=64,
            basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
            num_spherical=7, num_radial=6, envelope_exponent=5,
            num_before_skip=1, num_after_skip=1, num_output_layers=1):
        super(TdComNet, self).__init__()

        self.act = nn.SiLU()
        self.cutoff = cutoff
        self.init_e = init(num_radial, hidden_channels, use_node_features=False)
        self.init_v = update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers)
        self.init_u = update_u()

        self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)

        self.update_vs = torch.nn.ModuleList([
            update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers) for _ in
            range(num_layers)])

        self.update_es = torch.nn.ModuleList([
            update_e(hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion,
                     num_spherical, num_radial, num_before_skip, num_after_skip) for _ in range(num_layers)])

        self.update_us = torch.nn.ModuleList([update_u() for _ in range(num_layers)])

    def forward(self, batch_data):

        x_all, pos, batch = batch_data.x, batch_data.pos, batch_data.batch
        # edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        edge_index = batch_data.edge_index

        num_nodes = x_all.size(0)

        dist, angle, torsion, i, j, idx_kj, idx_ji, q_r = xyz_to_dat(pos, edge_index, num_nodes, num=-1, n=8,
                                                                     sort_by_angle=True)

        emb = self.emb(dist, angle, torsion, idx_kj)  # 1960*6  36014*42  36014*294

        e = self.init_e(x_all, emb, i, j)  # 1960*128  1960*128
        v = self.init_v(e, i)  # 这个有待修改  115*1
        u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0)), v, batch)  # 1*1

        for update_e, update_v, update_u in zip(self.update_es, self.update_vs, self.update_us):
            e = update_e(e, emb, idx_kj, idx_ji,q_r)
            v = update_v(e, i)
            u = update_u(u, v, batch)
        return u
