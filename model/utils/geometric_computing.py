# Based on the code from: https://github.com/klicperajo/dimenet,
# https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/models/dimenet.py
from torch_geometric.nn import radius_graph
import torch
from torch_scatter import scatter
from torch_sparse import SparseTensor
from math import pi as PI
import numpy as np

from model.utils.ops import quaternion_product

torch.set_printoptions(precision=4, sci_mode=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def generate_matrix(n):
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        matrix[i, :i + 1] = np.arange(1, i + 2)
    return matrix


def matrix_to_dict(matrix):
    n, m = matrix.shape
    return {i + 1: matrix[i, :].tolist() for i in range(n)}


def generate_matrix_tensor(n):
    matrix = torch.zeros((n, n), dtype=torch.int)
    for i in range(n):
        matrix[i, :i + 1] = torch.arange(1, i + 2)
    return matrix


def matrix_to_dict_tensor(matrix):
    n, m = matrix.shape
    return {torch.tensor(i + 1): matrix[i, :] for i in range(n)}


def node_edge_index(i, j, num_triplets, adj_t_row, kji=True, device='cpu'):
    # node index, k->j->i or j->i->m
    idx_i = i.repeat_interleave(num_triplets)
    idx_j = j.repeat_interleave(num_triplets)
    idx_k_or_m = adj_t_row.storage.col()
    if kji:
        mask = idx_i != idx_k_or_m
    else:
        mask = idx_j != idx_k_or_m
    idx_i, idx_j, idx_k_or_m = idx_i[mask], idx_j[mask], idx_k_or_m[mask]
    # edge index, kj->ji or ji->im
    idx_kj_or_im = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return idx_i, idx_j, idx_k_or_m, idx_kj_or_im, idx_ji, mask


def angle_compute(pos1, pos2, torsion, kji=True, device='cpu'):
    a = (pos1 * pos2).sum(dim=-1)  # cos(angle) * |pos_ji| * |pos_jk|, angle [0, 2pi]
    b = torch.cross(pos1, pos2).norm(dim=-1)  # sin(angle) * |pos_ji| * |pos_jk|, angle [0, pi]
    u = torch.cross(pos1, pos2)  # remove -0.0
    u_norm = u.norm(dim=-1, keepdim=True)  # [num_tri, 1]
    u = u / (u_norm + 1e-12)  # [num_tri, 3]
    index = torch.abs(u) < 1e-6  # 1) remove -0.0, 2) stable sort.
    u[index] = 0.0   # u是转动轴向量
    angle = torch.atan2(b, a)  # b = y, a =x, compute arctan(b/a)
    s = torch.cos(angle / 2).unsqueeze(-1)
    aa = angle.unsqueeze(-1).repeat(1, 3) / 2

    axis = u * torch.sin(angle.unsqueeze(-1).repeat(1, 3) / 2)  # 将结果张量沿着最后一个维度重复3次
    q1 = torch.cat([s, axis], dim=-1)  # [num_tri, 4]

    return angle, u, q1


def combination(angle, u, q, dist, device='cpu'):
    flag = torch.tensor([10] * q.size(0)).to(device)
    torsion = torch.tensor([0] * q.size(0)).to(device)
    cob_step1 = torch.cat(
        [q, u, flag.unsqueeze(dim=-1), dist.unsqueeze(dim=-1), angle.unsqueeze(dim=-1), torsion.unsqueeze(dim=-1)],
        dim=-1)

    return cob_step1


def map_padding(n, idx_ji, pos_edge, cob_step1, device='cpu'):
    dict_matrix = generate_matrix(n) - np.ones((n, n), dtype=int)
    dict = matrix_to_dict(dict_matrix)
    # idx_ji_count = pd.DataFrame(idx_ji.cpu()).value_counts(sort=False).values
    idx_ji_count = np.unique(idx_ji.cpu(), return_counts=True)[1]
    temp = np.zeros((idx_ji_count.shape[0], n), dtype=int)
    for key, value in dict.items():
        temp[idx_ji_count == key] = np.array(value)
    temp = temp.reshape(1, -1).squeeze(axis=0)
    x = idx_ji.cpu()  # 1st dim
    y = torch.tensor(np.delete(temp, np.where(temp == -1)))  # 2nd dim
    map = torch.stack((x, y), dim=1)
    cob_step2 = torch.zeros((pos_edge.size(0), n, 11)).to(device)
    cob_step2[:, :, 0] = 1
    cob_step2[:, :, -3] = 100
    cob_step2[:, :, -2] = 3 * PI
    cob_step2[map[:, 0], map[:, 1]] = cob_step1
    return cob_step2


def map_padding_tensor(n, idx_ji, pos_edge, cob_step1, device='cpu'):
    # n每个边的四元数数量 边索引  边向量  维度扩充后的
    dict_matrix = (generate_matrix_tensor(n) - torch.ones((n, n), dtype=int)).to(device)
    dict = matrix_to_dict_tensor(dict_matrix)
    _, idx_ji_count = idx_ji.unique(return_counts=True)
    temp = torch.zeros((idx_ji_count.shape[0], n), dtype=int).to(device)  # 边索引数*30
    for key, value in dict.items():
        temp[idx_ji_count == key] = value
    temp = temp.reshape(1, -1).squeeze(axis=0)
    mask = temp != -1
    y = temp[mask]
    x = idx_ji
    map = torch.stack((x, y), dim=1)   # x为边数
    cob_step2 = torch.zeros((pos_edge.size(0), n, 11)).to(device)
    cob_step2[:, :, 0] = 1
    cob_step2[:, :, -3] = 100
    cob_step2[:, :, -2] = 3 * PI
    ss = map[:, 0]
    ss = map[:, 1]
    cob_step2[map[:, 0], map[:, 1]] = cob_step1
    return cob_step2


def quaternions_sort(cob, pos_edge, dist, sort_by_angle=True, device='cpu'):
    if sort_by_angle:
        _, sort_idx = torch.sort(cob[..., -2], dim=-1, descending=False, stable=True)  # stable sort, sort_by_angle
        cob = cob[torch.arange(cob.shape[0])[:, None], sort_idx]
    else:
        _, sort_idx = torch.sort(cob[..., -3], dim=-1, descending=False, stable=True)  # stable sort, sort_by_dist
        cob = cob[torch.arange(cob.shape[0])[:, None], sort_idx]

    quaternion = cob[:, :, 0:4]

    return quaternion


def xyz_to_dat(pos, edge_index, num_nodes, n=30, num=-1, sort_by_angle=True, use_torsion=True):
    j, i = edge_index  # j-> i # [num_edge]
    device = j.device

    ## Calculate distance. # number of edges
    pos_edge = pos[i] - pos[j]
    dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()  # [num_edge]

    value = torch.arange(j.size(0), device=j.device)
    adj_t1 = SparseTensor(row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t2 = adj_t1.t()  # transpose
    adj_t_row1 = adj_t1[j]
    adj_t_row2 = adj_t2[i]
    num_triplets1 = adj_t_row1.set_value(None).sum(dim=1).to(torch.long)
    num_triplets2 = adj_t_row2.set_value(None).sum(dim=1).to(torch.long)

    # Node and edge index
    idx_i1, idx_j1, idx_k, idx_kj, idx_ji1, mask1 = node_edge_index(i, j, num_triplets1, adj_t_row1, kji=True,
                                                                    device=device)
    idx_i2, idx_j2, idx_m, idx_im, idx_ji2, mask2 = node_edge_index(i, j, num_triplets2, adj_t_row2, kji=False,
                                                                    device=device)

    ## Torsion  四个向量
    pos_ji = pos[idx_i1] - pos[idx_j1]  # p_ji [14, 3]
    pos_jk = pos[idx_k] - pos[idx_j1]  # p_jk
    pos_ij = pos[idx_j2] - pos[idx_i2]  # p_ij
    pos_im = pos[idx_m] - pos[idx_i2]  # p_im

    if use_torsion:
        idx_batch = torch.arange(len(idx_i1), device=device)
        idx_k_n = adj_t1[idx_j1].storage.col()
        repeat = num_triplets1
        num_triplets_t = num_triplets1.repeat_interleave(repeat)[mask1]
        idx_i_t = idx_i1.repeat_interleave(num_triplets_t)
        idx_j_t = idx_j1.repeat_interleave(num_triplets_t)
        idx_k_t = idx_k.repeat_interleave(num_triplets_t)
        idx_batch_t = idx_batch.repeat_interleave(num_triplets_t)
        mask = idx_i_t != idx_k_n
        idx_i_t, idx_j_t, idx_k_t, idx_k_n, idx_batch_t = idx_i_t[mask], idx_j_t[mask], idx_k_t[mask], idx_k_n[mask], \
            idx_batch_t[mask]
        pos_j0 = pos[idx_k_t] - pos[idx_j_t]
        pos_ji_torsion = pos[idx_i_t] - pos[idx_j_t]
        pos_jk_torsion = pos[idx_k_n] - pos[idx_j_t]
        dist_ji = pos_ji_torsion.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(pos_ji_torsion, pos_j0)
        plane2 = torch.cross(pos_ji_torsion, pos_jk_torsion)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji_torsion).sum(dim=-1) / dist_ji
        torsion1 = torch.atan2(b, a)  # -pi to pi
        torsion1[torsion1 <= 0] = torsion1[torsion1 <= 0] + 2 * PI  # 0 to 2pi
        torsion = scatter(torsion1, idx_batch_t, reduce='min')
    else:
        torsion = None
    # print(torsion)
    # Angle-------------rotation axis转动轴向量---------------quaterion四元数
    angle_kji, u_kji, q_kji = angle_compute(pos_jk, pos_ji, torsion, kji=True, device=device)
    angle_jim, u_jim, q_jim = angle_compute(pos_ij, pos_im, torsion, kji=False, device=device)

    ## dist
    dist_kj = dist[idx_kj]
    dist_im = dist[idx_im]

    # ## Combine all the information
    cob_kji_step1 = combination(angle_kji, u_kji, q_kji, dist_kj, device=device)
    cob_jim_step1 = combination(angle_jim, u_jim, q_jim, dist_im, device=device)

    ## Map and padding according to j->i [num_tri, 11] ->[num_edge, 30, 11]
    # cob_kji_step2 = map_padding(n, idx_ji1, pos_edge, cob_kji_step1, device)
    # cob_jim_step2 = map_padding(n, idx_ji2, pos_edge, cob_jim_step1, device)
    cob_kji_step2 = map_padding_tensor(n, idx_ji1, pos_edge, cob_kji_step1, device)
    cob_jim_step2 = map_padding_tensor(n, idx_ji2, pos_edge, cob_jim_step1, device)

    ## sort each side
    cob_kji_step2 = quaternions_sort(cob_kji_step2, pos_edge, dist, sort_by_angle=sort_by_angle, device=device)
    cob_jim_step2 = quaternions_sort(cob_jim_step2, pos_edge, dist, sort_by_angle=sort_by_angle, device=device)

    ## Combine quaternions in the two sides
    if num == -1:
        cob = torch.cat([cob_kji_step2, cob_jim_step2], dim=1)
    else:
        cob = torch.cat([cob_kji_step2[:, :num], cob_jim_step2[:, :num]], dim=1)

    # cob = cob_jim_step2 # w/o j side quaternions
    # cob = cob_kji_step2 # w/o i side quaternions
    # quaternion = cob[:, :, 0:4] # w/o sorting

    ## Sort quaternion
    quaternion = quaternions_sort(cob, pos_edge, dist, sort_by_angle=sort_by_angle, device=device)
    q_r, q_i, q_j, q_k = quaternion_product(quaternion)

    # q_r = 0

    return dist, angle_kji, torsion, i, j, idx_kj, idx_ji1, q_r


if __name__ == '__main__':
    """
    adj_t
    [[ 0,  0,  1,  0,  0,  0,  0,  0],
    [ 2,  0,  0,  0,  0,  0,  0,  0],
    [ 3,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  4,  5,  0,  6],
    [ 0,  0,  0,  7,  0,  0,  8,  0],
    [ 0,  0,  0,  9,  0,  0, 10,  0],
    [ 0,  0,  0,  0, 11, 12,  0,  0],
    [ 0,  0,  0, 13,  0,  0,  0,  0]]) # [8, 8]
    idx_k   [2,  1,  6,  6,  5,  7,  5,  4,  7,  4,  3,  3,  4,  5])
    idx_j1  [0,  0,  4,  5,  3,  3,  6,  3,  3,  6,  4,  5,  3,  3])
    idx_i1  [1,  2,  3,  3,  4,  4,  4,  5,  5,  5,  6,  6,  7,  7])
    idx_kj  [1,  0,  8,  10, 5,  6,  12, 4,  6,  11, 7,  9,  4,  5])
    idx_ji1 [2,  3,  4,  5,  7,  7,  8,  9,  9,  10, 11, 12, 13, 13])
================================================================================
    idx_j2  [1,  2,  4,  4,  5,  5,  7,  7,  3,  6,  3,  6,  4,  5])
    idx_i2  [0,  0,  3,  3,  3,  3,  3,  3,  4,  4,  5,  5,  6,  6])
    idx_m   [2,  1,  5,  7,  4,  7,  4,  5,  6,  3,  6,  3,  5,  4])
    idx_ji2 [0,  1,  4,  4,  5,  5,  6,  6,  7,  8,  9,  10, 11, 12])
    idx_im  [3,  2,  9,  13, 7,  13, 7,  9,  11, 4,  12, 5,  10, 8])
    
    num: Top number of quaternions are selected.
    n: The number quaternions for each edge, which is larger than max degree of atoms in dataset
    
    """
    import datetime

    starttime = datetime.datetime.now()
    # pos = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 2]], dtype=torch.float)
    # pos = torch.tensor([[0, 0, 0], [1, 0, 0], [0, -1, 0], [0, 0, 1], [1, 0, 1], [0, -1, 1], [1, -1, 1], [0, 0, 2]], dtype=torch.float)
    pos = torch.tensor(
        [[0.2, 0.4, 0.1], [1.2, 0.3, 0.4], [0.2, 0.6, 1], [1.5, 0.5, 0.5], [1, 1.2, 0], [1, -0.5, -0.866],
         [1, -0.866, -0.5]]).to(device)

    # pos = torch.tensor([[0.2, -0.4, 0.1], [1.2, -0.3, 0.4], [0.2, -0.6, 1], [1.5, -0.5, 0.5], [1, -1.2, 0], [1, 0.5, -0.866], [1, 0.866, -0.5]])
    # batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
    batch = torch.tensor([0, 0, 0, 0, 0, 0,0]).to(device)
    num_nodes = batch.size(0)
    edge_index = radius_graph(pos, r=1.5, batch=batch)
    j, i = edge_index
    dis = torch.norm((pos[i] - pos[j]), dim=1)
    dist, angle_kji, torsion, i, j, idx_kj, idx_ji1, q_r = xyz_to_dat(pos, edge_index, num_nodes, n=5, num=2,
                                                                      sort_by_angle=True)
    print(q_r)
    endtime = datetime.datetime.now()
    print(endtime - starttime)
