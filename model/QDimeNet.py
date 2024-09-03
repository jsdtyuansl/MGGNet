import torch
from torch import nn
from torch.nn import Linear, Embedding
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
from math import sqrt

from .utils.geometric_computing import xyz_to_dat
from .features import dist_emb, angle_emb


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


class init(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, act=swish):
        super(init, self).__init__()
        self.act = act
        self.emb = Embedding(95, hidden_channels)
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)

    def forward(self, x, emb, i, j):
        rbf, _ = emb
        x = self.emb(x)
        rbf0 = self.act(self.lin_rbf_0(rbf))
        e1 = self.act(self.lin(torch.cat([x[i], x[j], rbf0], dim=-1)))
        e2 = self.lin_rbf_1(rbf) * e1

        return e1, e2


class update_e(torch.nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size, num_spherical, num_radial,
                 num_before_skip, num_after_skip, act=swish):
        super(update_e, self).__init__()
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        # self.lin_cat = Linear(hidden_channels + 1, hidden_channels, bias=False)
        self.lin_cat = Linear(hidden_channels, hidden_channels, bias=False)

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_after_skip)
        ])

    def forward(self, x, emb, idx_kj, idx_ji, q_r):
        rbf0, sbf = emb
        x1, _ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_ji + x_kj

        e1 = self.act(self.lin_cat(e1))
        # e1 = self.act(self.lin_cat(torch.cat([e1, q_r], dim=-1)))

        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1

        return e1, e2


class update_v(torch.nn.Module):
    def __init__(self, hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init):
        super(update_v, self).__init__()
        self.act = act
        self.output_init = output_init

        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

    def forward(self, e, i):
        _, e2 = e
        v = scatter(e2, i, dim=0)
        v = self.lin_up(v)
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


class Q_DimeNetPP(torch.nn.Module):
    def __init__(
            self, energy_and_force=False, cutoff=5.0, num_layers=4,
            hidden_channels=128, out_channels=1, int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
            num_spherical=7, num_radial=6, envelope_exponent=5,
            num_before_skip=1, num_after_skip=2, num_output_layers=3,
            act=swish, output_init='GlorotOrthogonal', num=-1, n=30, sort_by_angle=True):
        super(Q_DimeNetPP, self).__init__()

        self.cutoff = cutoff
        self.energy_and_force = energy_and_force
        self.n = n
        self.sort_by_angle = sort_by_angle
        self.num = num

        self.init_e = init(num_radial, hidden_channels, act)
        self.init_v = update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init)
        self.init_u = update_u()
        self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)

        self.update_vs = torch.nn.ModuleList([
            update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init) for _ in
            range(num_layers)])

        self.update_es = torch.nn.ModuleList([
            update_e(
                hidden_channels, int_emb_size, basis_emb_size,
                num_spherical, num_radial,
                num_before_skip, num_after_skip,
                act,
            )
            for _ in range(num_layers)
        ])

        self.update_us = torch.nn.ModuleList([update_u() for _ in range(num_layers)])

        self.lin_node = Linear(48, hidden_channels, bias=False)
        self.lin_edge = Linear(12, hidden_channels, bias=False)

    def forward(self, data):
        batch = data.batch
        x = data.x
        pos = data.pos
        num_nodes = x.size(0)
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        dist, angle, torsion, i, j, idx_kj, idx_ji, q_r = xyz_to_dat(pos, edge_index, num_nodes, num=self.num, n=self.n,
                                                                     sort_by_angle=True,
                                                                     use_torsion=False)
        emb = self.emb(dist, angle, idx_kj)

        # Initialize edge, node, graph features
        e = self.lin_edge(edge_attr)
        e = (e, e)
        v = self.lin_node(x)

        u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0)), v, batch)  # scatter(v, batch, dim=0)

        for update_e, update_v, update_u in zip(self.update_es, self.update_vs, self.update_us):
            e = update_e(e, emb, idx_kj, idx_ji, q_r)
            v = update_v(e, i)
            u = update_u(u, v, batch)

        return u
