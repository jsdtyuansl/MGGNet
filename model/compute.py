# Based on the code from: https://github.com/klicperajo/dimenet,
# https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/models/dimenet.py

import torch
from torch_scatter import scatter
from torch_sparse import SparseTensor
from math import pi as PI

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def xyz_to_dat(pos, edge_index, num_nodes, use_torsion=True):
    j, i = edge_index  # j->i

    # Calculate distances. # number of edges
    dist = (pos[i] - pos[j]).norm(dim=1)  # 根据边索引计算距离向量

    value = torch.arange(j.size(0), device=device)

    adj_t = SparseTensor(row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes))
    # print(adj_t.to_dense())
    adj_t_row = adj_t[j]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)  # 每个边对应的三联体数量(大小为Ne的tensor)2

    # Node indices (k->j->i) for triplets.
    idx_i = i.repeat_interleave(num_triplets)
    idx_j = j.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
    # idx长度为三联体去重后的数量

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    # Calculate angles. 0 to pi求0到派之间的角度
    pos_ji = pos[idx_i] - pos[idx_j]
    pos_jk = pos[idx_k] - pos[idx_j]
    a = (pos_ji * pos_jk).sum(dim=-1)  # 向量点乘
    b = torch.cross(pos_ji, pos_jk).norm(dim=-1)  # 向量叉乘后取模
    angle = torch.atan2(b, a)

    if use_torsion:
        # Prepare torsion idx
        idx_batch = torch.arange(len(idx_i), device=device)
        idx_k_n = adj_t[idx_j].storage.col()
        repeat = num_triplets
        num_triplets_t = num_triplets.repeat_interleave(repeat)[mask]
        idx_i_t = idx_i.repeat_interleave(num_triplets_t)
        idx_j_t = idx_j.repeat_interleave(num_triplets_t)
        idx_k_t = idx_k.repeat_interleave(num_triplets_t)
        idx_batch_t = idx_batch.repeat_interleave(num_triplets_t)
        mask = idx_i_t != idx_k_n
        idx_i_t, idx_j_t, idx_k_t, idx_k_n, idx_batch_t = idx_i_t[mask], idx_j_t[mask], idx_k_t[mask], idx_k_n[mask], \
        idx_batch_t[mask]

        # Calculate torsions.
        pos_j0 = pos[idx_k_t] - pos[idx_j_t]
        pos_ji = pos[idx_i_t] - pos[idx_j_t]
        pos_jk = pos[idx_k_n] - pos[idx_j_t]
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(pos_ji, pos_j0)
        plane2 = torch.cross(pos_ji, pos_jk)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        torsion1 = torch.atan2(b, a)  # -pi to pi
        torsion1[torsion1 <= 0] += 2 * PI  # 0 to 2pi
        torsion = scatter(torsion1, idx_batch_t, reduce='min')

        return dist, angle, torsion, i, j, idx_kj, idx_ji

    else:
        return dist, angle, i, j, idx_kj, idx_ji
