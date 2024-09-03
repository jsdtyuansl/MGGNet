from torch_geometric.nn import GraphConv, MessagePassing, radius_graph, DimeNetPlusPlus
from torch_geometric.nn.norm import GraphNorm
from torch_scatter import scatter, scatter_min
import torch
from torch import nn
from torch import Tensor
from torch.nn import Linear, Dropout
import math
from math import sqrt
from .computecom import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class InteractionLayer(MessagePassing):
    def __init__(self, hidden_channels, middle_channels, num_radial, num_spherical, num_output_layers, output_channels,
                 edge_dim=12):
        super(InteractionLayer, self).__init__()
        self.act = nn.SiLU()

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, hidden_channels)
        # self.edgelin = Linear(edge_dim, hidden_channels)

        self.lin_cat = Linear(3 * hidden_channels, hidden_channels)

        self.norm = GraphNorm(hidden_channels)

        # tbf 3*2^2  sbf 3*2  贝塞尔和球谐函数的维度变换
        self.lin_rbf = nn.Sequential(Linear(9, hidden_channels), nn.SiLU(), Linear(hidden_channels, hidden_channels),
                                     nn.SiLU())
        self.lin_s = nn.Sequential(Linear(num_radial * num_spherical, hidden_channels), nn.SiLU(),
                                   Linear(hidden_channels, hidden_channels), nn.SiLU())
        self.lin_t = nn.Sequential(Linear(num_radial * num_spherical ** 2, hidden_channels), nn.SiLU(),
                                   Linear(hidden_channels, hidden_channels), nn.SiLU())
        self.lin_pos = nn.Sequential(Linear(16, hidden_channels), nn.SiLU(), Linear(hidden_channels, hidden_channels),
                                     nn.SiLU())

        self.out1 = nn.Sequential(Linear(hidden_channels, hidden_channels),
                                  Dropout(0.1),
                                  nn.LeakyReLU(),
                                  nn.BatchNorm1d(hidden_channels))
        self.out2 = nn.Sequential(Linear(hidden_channels, hidden_channels),
                                  Dropout(0.1),
                                  nn.LeakyReLU(),
                                  nn.BatchNorm1d(hidden_channels))
        self.out3 = nn.Sequential(Linear(hidden_channels, hidden_channels),
                                  Dropout(0.1),
                                  nn.LeakyReLU(),
                                  nn.BatchNorm1d(hidden_channels))

    def forward(self, x, pos, feature1, feature2, feature3, edge_index, edge_attr, batch):
        # rvf tbf sbf
        rbf = self.lin_rbf(feature1)
        h1 = self.propagate(edge_index, x=x, edge_attr=edge_attr, base=rbf)
        h1 = self.act(self.lin1(h1+x))
        out1 = self.out1(h1)

        tbf = self.lin_t(feature2)
        h2 = self.propagate(edge_index, x=x, edge_attr=edge_attr, base=tbf)
        h2 = self.act(self.lin2(h2+x))
        out2 = self.out2(h2)

        sbf = self.lin_s(feature3)
        h3 = self.propagate(edge_index, x=x, edge_attr=edge_attr, base=sbf)
        h3 = self.act(self.lin3(h3+x))
        out3 = self.out3(h3)

        h = self.lin_cat(torch.cat([out1, out2, out3], 1))
        return h

    def message(self, x_j: Tensor, edge_attr: Tensor, base: Tensor):
        return self.act(x_j * base)


class HomoNet2(nn.Module):
    def __init__(
            self, in_channels=48, cutoff=5.0, num_layers=3, hidden_channels=256, middle_channels=64, out_channels=1,
            num_radial=6,
            num_spherical=7,
            num_output_layers=2,
    ):
        super(HomoNet2, self).__init__()
        self.out_channels = out_channels
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.act = nn.SiLU()

        self.tbf = torsion_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.sbf = angle_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)

        self.line_node = nn.Sequential(Linear(in_channels, hidden_channels), nn.SiLU())

        self.interaction_layers = torch.nn.ModuleList(
            [
                InteractionLayer(
                    hidden_channels,
                    middle_channels,
                    num_radial,
                    num_spherical,
                    num_output_layers,
                    hidden_channels,
                    edge_dim=12
                )
                for _ in range(num_layers)
            ]
        )
        self.edge_lin = Linear(12, hidden_channels)
        self.edge_mlp = Linear(2 * hidden_channels, hidden_channels)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lin_out = Linear(hidden_channels, out_channels)

    def rbf(self, D, D_min, D_max, D_count):
        D_mu = torch.linspace(D_min, D_max, D_count).to('cuda')
        D_mu = D_mu.view([1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def pos_emb(self, edge_index, num_pos_emb=16):
        d = edge_index[0] - edge_index[1]
        frequency = torch.exp(
            torch.arange(0, num_pos_emb, 2, dtype=torch.float32, device=edge_index.device)
            * -(np.log(10000.0) / num_pos_emb)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def forward(self, data):
        x, pos, edge_index, edge_attr, batch = data.x, data.pos, data.edge_index, data.edge_attr, data.batch
        num_nodes = x.size(0)
        j, i = edge_index
        pos_emb = self.pos_emb(edge_index, 16)

        vecs = pos[j] - pos[i]
        dist = vecs.norm(dim=-1)  # 原子距离
        rbf = self.rbf(dist, 0., 6., 9)

        x = self.line_node(x)
        # Calculate distances.
        _, argmin0 = scatter_min(dist, i, dim_size=num_nodes)  # 挑选最近的节点在j中的索引索引位置

        argmin0[argmin0 >= len(i)] = 0
        n0 = j[argmin0]  # 由j得到i中各原子最近邻的原子的排列号
        add = torch.zeros_like(dist).to(dist.device)
        add[argmin0] = self.cutoff
        dist1 = dist + add  # 扩大距离方便求次邻居

        _, argmin1 = scatter_min(dist1, i, dim_size=num_nodes)
        argmin1[argmin1 >= len(i)] = 0
        n1 = j[argmin1]  # 得到原子i中第二紧邻的原子编号
        # --------------------------------------------------------

        _, argmin0_j = scatter_min(dist, j, dim_size=num_nodes)
        argmin0_j[argmin0_j >= len(j)] = 0
        n0_j = i[argmin0_j]  # 得到j的最近邻居

        add_j = torch.zeros_like(dist).to(dist.device)
        add_j[argmin0_j] = self.cutoff
        dist1_j = dist + add_j

        # i[argmin] = range(0, num_nodes)
        _, argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)
        argmin1_j[argmin1_j >= len(j)] = 0
        n1_j = i[argmin1_j]  # 得到j的次近邻居

        # ----------------------------------------------------------
        # n0, n1 for i  根据edge_index，根据边得到target的最近邻居
        n0 = n0[i]
        n1 = n1[i]

        # n0, n1 for j
        n0_j = n0_j[j]
        n1_j = n1_j[j]

        # tau: (iref, i, j, jref)
        # 如果i的最近邻居为j，那选i的次近邻
        # 如果j的最近邻居为i，那选j的次近邻
        mask_iref = n0 == j
        iref = torch.clone(n0)
        iref[mask_iref] = n1[mask_iref]
        idx_iref = argmin0[i]
        idx_iref[mask_iref] = argmin1[i][mask_iref]

        mask_jref = n0_j == i
        jref = torch.clone(n0_j)
        jref[mask_jref] = n1_j[mask_jref]
        idx_jref = argmin0_j[j]
        idx_jref[mask_jref] = argmin1_j[j][mask_jref]

        pos_ji, pos_in0, pos_in1, pos_iref, pos_jref_j = (
            vecs,
            vecs[argmin0][i],
            vecs[argmin1][i],
            vecs[idx_iref],
            vecs[idx_jref]
        )

        # 计算角度
        a = ((-pos_ji) * pos_in0).sum(dim=-1)
        b = torch.cross(-pos_ji, pos_in0).norm(dim=-1)
        theta = torch.atan2(b, a)
        theta[theta < 0] = theta[theta < 0] + math.pi

        # 计算二面角
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(-pos_ji, pos_in0)
        plane2 = torch.cross(-pos_ji, pos_in1)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        phi = torch.atan2(b, a)
        phi[phi < 0] = phi[phi < 0] + math.pi

        # 计算旋转
        plane1 = torch.cross(pos_ji, pos_jref_j)
        plane2 = torch.cross(pos_ji, pos_iref)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        tau = torch.atan2(b, a)
        tau[tau < 0] = tau[tau < 0] + math.pi

        tbf = self.tbf(dist, theta, phi)  # Ne*12
        sbf = self.sbf(dist, tau)  # Ne*6

        for interaction_layer in self.interaction_layers:
            x = interaction_layer(x, pos_emb, rbf, tbf, sbf, edge_index, edge_attr, batch)

        for lin in self.lins:
            x = self.act(lin(x))

        x = self.lin_out(x)

        out = scatter(x, batch, dim=0)
        return out
