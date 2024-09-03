from torch_geometric.nn import GraphConv, MessagePassing, radius_graph
from torch_geometric.nn.norm import GraphNorm
from torch_scatter import scatter, scatter_min
import torch
from torch import nn
from torch import Tensor
from torch.nn import Linear, Dropout
import math
import numpy as np
from .utils.geometric_computing import xyz_to_dat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class InteractionLayer(MessagePassing):
    def __init__(self, hidden_channels, middle_channels, num_radial, num_spherical, num_output_layers, output_channels,
                 edge_dim=12):
        super(InteractionLayer, self).__init__(aggr = 'mean')
        self.act = nn.SiLU()

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, hidden_channels)

        self.lin_cat = Linear(hidden_channels*2, hidden_channels)

        self.norm = GraphNorm(hidden_channels)

        # tbf 3*2^2  sbf 3*2  贝塞尔和球谐函数的维度变换
        self.lin_t = nn.Sequential(Linear(num_radial * num_spherical ** 2, hidden_channels, bias=False), nn.SiLU())
        self.lin_s = nn.Sequential(Linear(num_radial * num_spherical, hidden_channels, bias=False), nn.SiLU())
        self.lin_rbf = nn.Sequential(Linear(9, hidden_channels, bias=False), nn.SiLU())
        self.lin_qr = nn.Sequential(Linear(1, hidden_channels, bias=False), nn.SiLU())
        self.lin_pos = nn.Sequential(Linear(16, hidden_channels, bias=False), nn.SiLU())
        
        self.lin_res= nn.Linear(hidden_channels,hidden_channels)

        self.out = nn.Sequential(Linear(hidden_channels, hidden_channels),
                                 Dropout(0.1),
                                 nn.LeakyReLU(),
                                 nn.BatchNorm1d(hidden_channels))

    def forward(self, x, pos, rbf, q_r, edge_index, edge_attr, batch):
        rbf = self.lin_rbf(rbf)
        q_r = self.lin_qr(q_r)
        base = self.lin_cat(torch.cat([rbf, q_r], dim=-1))
        
        h1 = self.propagate(edge_index, x=x, edge_attr=edge_attr, base=base)
        h1 = self.lin2(h1)
        h1 = self.act(h1)
  
        h = self.out(h1 + x)

        return h

    def message(self, x_j: Tensor, edge_attr: Tensor, base: Tensor):
        return x_j * base


class HomoNetQM(nn.Module):
    def __init__(
            self, in_channels=48, cutoff=5.0, num_layers=3, hidden_channels=256, middle_channels=64, out_channels=1,
            num_radial=6,
            num_spherical=7,
            num_output_layers=2,
    ):
        super(HomoNetQM, self).__init__()
        self.out_channels = out_channels
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.act = nn.SiLU()

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
        self.edgelin = Linear(12, hidden_channels)
        self.edge_mlp = Linear(2 * hidden_channels, hidden_channels)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lin_out = Linear(hidden_channels, out_channels)

    def rbf(self, D, D_min, D_max, D_count):
        D_mu = torch.linspace(D_min, D_max, D_count).to('cuda')  # 生成等间隔张量
        D_mu = D_mu.view([1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def pos_emb(self, edge_index, num_pos_emb=16):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_pos_emb, 2, dtype=torch.float32, device=edge_index.device)
            * -(np.log(10000.0) / num_pos_emb)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def forward(self, data):
        batch = data.batch
        x = data.x
        pos = data.pos
        num_nodes = x.size(0)

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        pos_emb = self.pos_emb(edge_index, 16)

        # edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        j, i = edge_index

        vecs = pos[j] - pos[i]
        dist = vecs.norm(dim=-1)  # 原子距离

        rbf = self.rbf(dist, 0., 6., 9)
        dist, angle, torsion, i, j, idx_kj, idx_ji, q_r = xyz_to_dat(pos, edge_index, num_nodes, num=-1, n=30,
                                                                     sort_by_angle=True, use_torsion=False)

        x = self.line_node(x)

        for interaction_layer in self.interaction_layers:
            x = interaction_layer(x, pos_emb, rbf, q_r, edge_index, edge_attr, batch)

        for lin in self.lins:
            x = self.act(lin(x))

        x = self.lin_out(x)

        out = scatter(x, batch, dim=0)
        return out
