import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, CGConv, HeteroConv, Linear, SAGEConv, GATConv, global_max_pool, MLP, \
    AttentiveFP, global_mean_pool, BatchNorm, HANConv, HGTConv, GCN, HEATConv
from torch_scatter import scatter_mean, scatter_sum, scatter_max
from torch_geometric.nn.conv import MessagePassing


class HeteroNet(torch.nn.Module):
    def __init__(self, edge_dim, hidden_channels, out_channels):
        super().__init__()
        self.metadata = [('ligand', 'to', 'protein'), ('protein', 'rev_to', 'ligand')]
        self.node_lin = Linear(in_channels=44, out_channels=hidden_channels, bias=False)
        self.edge_lin = Linear(in_channels=1, out_channels=8, bias=False)
        self.edge_mlp = MLP(channel_list=[hidden_channels * 2 + 8, 512, 128], dropout=0.1)
        self.lin_mpl = Linear(in_channels=128, out_channels=out_channels // 2)

        self.conv_1 = HeteroConv(
            {
                edge_type: NConv(hidden_channels, hidden_channels) for edge_type in self.metadata
            }
        )
        self.conv_2 = HeteroConv(
            {
                edge_type: NConv(hidden_channels, hidden_channels) for edge_type in self.metadata
            }
        )
        self.conv_3 = HeteroConv(
            {
                # ///123
                edge_type: NConv(hidden_channels, hidden_channels) for edge_type in self.metadata
            }
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch_dict):
        x_dict = {
            'ligand': self.node_lin(x_dict['ligand']),
            'protein': self.node_lin(x_dict['protein'])
        }

        x1_dict = self.conv_1(x_dict, edge_index_dict, edge_attr_dict)
        x1_dict = {key: F.leaky_relu(x) for key, x in x1_dict.items()}

        x2_dict = self.conv_2(x1_dict, edge_index_dict, edge_attr_dict)
        x2_dict = {key: F.leaky_relu(x) for key, x in x2_dict.items()}

        x3_dict = self.conv_3(x2_dict, edge_index_dict, edge_attr_dict)
        x3_dict = {key: F.leaky_relu(x) for key, x in x3_dict.items()}

        x_dict['ligand'] = x1_dict['ligand'] + x2_dict['ligand'] + x3_dict[
            'ligand']

        x_dict['protein'] = x1_dict['protein'] + x2_dict['protein'] + x3_dict[
            'protein']

        src, tar = edge_index_dict[('ligand', 'to', 'protein')]
        edge_repr = torch.cat([x_dict['ligand'][src], x_dict['protein'][tar]],
                              dim=-1)

        d_pl = self.edge_lin(edge_attr_dict[('ligand', 'to', 'protein')])
        edge_repr = torch.cat((edge_repr, d_pl), dim=1)

        m_pl = self.edge_mlp(edge_repr)
        edge_batch = batch_dict['ligand'][src]
        w_pl = torch.tanh(self.lin_mpl(m_pl))
        m_w = w_pl * m_pl
        m_w = scatter_sum(m_w, edge_batch, dim=0)
        m_max, _ = scatter_max(m_pl, edge_batch, dim=0)
        m_out = torch.cat((m_w, m_max), dim=1)
        return m_out


class NConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, bias=True):
        super(NConv, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.fc_self = nn.Linear(in_channels, out_channels, bias=False)

        self.lin_base = nn.Sequential(nn.Linear(8, in_channels), nn.SiLU())

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)

    def forward(self, x, edge_index, edge_attr):
        base = self.lin_base(_rbf(edge_attr, D_min=0., D_max=5., D_count=8))

        out = self.propagate(edge_index, x=x, base=base)
        out = self.fc_self(x[1]) + out
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, base):
        return self.fc_neigh(base * x_j)


def _rbf(D, D_min=0., D_max=5., D_count=8, device='cuda'):
    D = torch.squeeze(D)
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF
