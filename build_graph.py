import os
import pandas as pd
import numpy as np
import pickle
from scipy.spatial import distance_matrix
import torch
from tqdm import tqdm
from torch_geometric.data import Data, HeteroData
from rdkit import Chem
import warnings
warnings.filterwarnings('ignore')
import torch_geometric.transforms as T


def get_atom_fea(atom, explicit_H=True):  # 原子特征
    results = one_encoding_symbol(atom.GetAtomicNum(),
                                  ['B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogens', 'metals', 'others']) + \
              one_encoding(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3] + ['others']) + \
              one_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + \
              one_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              one_encoding(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                           SP3D, Chem.rdchem.HybridizationType.SP3D2
              ] + ['others']) + [int(atom.GetIsAromatic())]
    if explicit_H:
        results = results + one_encoding(atom.GetTotalNumHs(),
                                         [0, 1, 2, 3, 4])
    return results


def get_bond_fea(bond):  # 12维边特征
    results = one_encoding(bond.GetBondType(), [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                                Chem.rdchem.BondType.TRIPLE,
                                                Chem.rdchem.BondType.AROMATIC, 'others']) + \
              one_encoding(bond.GetStereo(),
                           [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
                            Chem.rdchem.BondStereo.STEREOZ,
                            Chem.rdchem.BondStereo.STEREOE, 'others']) + \
              [int(bond.IsInRing())] + \
              [int(bond.GetIsConjugated())]
    return results


def one_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]  # 不知道的类型
    return list(map(lambda s: int(x == s), allowable_set))


def one_encoding_symbol(x, allowable_set):
    metals = [3, 4, 11, 12, 13] + list(range(19, 33)) + list(range(37, 52)) + list(range(55, 85))

    atom_classes = [([5], 'B'), ([6], 'C'), ([7], 'N'), ([8], 'O'), ([15], 'P'), ([16], 'S'), ([34], 'Se'),
                    ([9, 17, 35, 53], 'halogens'), (metals, 'metals')]

    for num, atom_class in atom_classes:
        if x in num:
            return one_encoding(atom_class, allowable_set)
    return one_encoding('others', allowable_set)


def get_lig_features(ligand):
    ligand_positions = ligand.GetConformer().GetPositions()  # 氢原子之外的原子坐标
    nligand_atoms = len(ligand.GetAtoms())
    atom_features_list = []
    bond_features_list = []
    edges_list = []
    pos_list = []

    for i in range(nligand_atoms):
        atom = ligand.GetAtomWithIdx(int(i))

        atom_feature = list(get_atom_fea(atom, explicit_H=True))
        atom_features_list.append(atom_feature)
        pos_list.append(ligand_positions[i].tolist())

    for bond in ligand.GetBonds():
        i = bond.GetBeginAtomIdx()  # 文件中的索引从1开始，打印出来的索引从0开始的
        j = bond.GetEndAtomIdx()
        bond_feature = get_bond_fea(bond)

        edges_list.append((i, j))
        bond_features_list.append(bond_feature)
        edges_list.append((j, i))
        bond_features_list.append(bond_feature)

    return torch.FloatTensor(np.array(atom_features_list)), torch.tensor(np.array(edges_list),
                                                                         dtype=torch.long).T, torch.FloatTensor(
        np.array(bond_features_list)), torch.FloatTensor(np.array(pos_list))


def get_pocket_features(pocket):
    pocket_positions = pocket.GetConformer().GetPositions()  # 氢原子之外的原子坐标
    npocket_atoms = len(pocket.GetAtoms())
    atom_features_list = []
    bond_features_list = []
    edges_list = []
    pos_list = []

    for i in range(npocket_atoms):
        atom = pocket.GetAtomWithIdx(int(i))
        atom_feature = list(
            get_atom_fea(atom, explicit_H=True))  # + pocket_positions[i].tolist()
        atom_features_list.append(atom_feature)
        pos_list.append(pocket_positions[i].tolist())

    # print(len(ligand.GetBonds()))

    for bond in pocket.GetBonds():
        i = bond.GetBeginAtomIdx()  # 文件中的索引从1开始，实际索引从0开始的
        j = bond.GetEndAtomIdx()
        bond_feature = get_bond_fea(bond)

        edges_list.append((i, j))
        bond_features_list.append(bond_feature)
        edges_list.append((j, i))
        bond_features_list.append(bond_feature)

    return torch.FloatTensor(np.array(atom_features_list)), torch.tensor(np.array(edges_list),
                                                                         dtype=torch.long).T, torch.FloatTensor(
        np.array(bond_features_list)), torch.FloatTensor(np.array(pos_list))


# def get_com_features(ligand, pocket, cut=5.):
#     ligand_positions = ligand.GetConformer().GetPositions()
#     pocket_positions = pocket.GetConformer().GetPositions()
#     atom_num_l = ligand.GetNumAtoms()
#
#     dis_matrix = distance_matrix(ligand_positions, pocket_positions)  # 距离矩阵
#     node_idx = np.where(dis_matrix < cut)  # 距离阈值筛选,返回节点索引
#
#     pos_list = []
#     idx_list = [[], []]  # 整合起来的索引，待重排
#     x_all = []
#     edge_index = []
#
#     for i in node_idx[0]:
#         if i not in idx_list[0]:
#             c_lig_fea = get_atom_fea(ligand.GetAtomWithIdx(int(i)), explicit_H=True)
#             x_all.append(c_lig_fea)
#             pos_list.append(ligand_positions[i].tolist())
#             idx_list[0].append(i)  # 配体原子的单独索引
#
#     for j in node_idx[1]:
#         if j + atom_num_l not in idx_list[1]:
#             c_poc_fea = get_atom_fea(pocket.GetAtomWithIdx(int(j)), explicit_H=True)
#             x_all.append(c_poc_fea)
#             pos_list.append(pocket_positions[j].tolist())
#             idx_list[1].append(j + atom_num_l)  # 口袋原子的单独索引
#
#     id = idx_list[0] + idx_list[1]
#
#     for i, j in zip(node_idx[0], node_idx[1]):
#         edge_index.append([id.index(i), id.index(j + atom_num_l)])
#         edge_index.append([id.index(j + atom_num_l), id.index(i)])
#
#     edge_index = torch.LongTensor(edge_index).t().contiguous()
#
#     return torch.FloatTensor(np.array(x_all)), torch.FloatTensor(np.array(pos_list)), edge_index


def get_bigraph(ligand, pocket, cut=5.):
    # 二分图分两种节点
    ligand_positions = ligand.GetConformer().GetPositions()
    pocket_positions = pocket.GetConformer().GetPositions()
    atom_num_l = ligand.GetNumAtoms()

    dis_matrix = distance_matrix(ligand_positions, pocket_positions)  # 距离矩阵
    node_idx = np.where(dis_matrix < cut)  # 距离阈值筛选,返回节点索引

    # pos_list = []
    idx_list = [[], []]  # 整合的索引，待重排
    x_all = []
    edge_index = []
    edge_attr = []

    for i in node_idx[0]:
        if i not in idx_list[0]:
            c_lig_fea = get_atom_fea(ligand.GetAtomWithIdx(int(i)), explicit_H=True)
            x_all.append(c_lig_fea)
            idx_list[0].append(i)  # 配体原子的单独索引

    for j in node_idx[1]:
        if j not in idx_list[1]:
            c_poc_fea = get_atom_fea(pocket.GetAtomWithIdx(int(j)), explicit_H=True)
            x_all.append(c_poc_fea)
            idx_list[1].append(j)  # 口袋原子的单独索引

    for i, j in zip(node_idx[0], node_idx[1]):
        edge_attr.append([dis_matrix[i][j]])
        edge_index.append([idx_list[0].index(i), idx_list[1].index(j)])

    bi_graph = HeteroData()

    bi_graph['ligand'].x = torch.FloatTensor(x_all[:len(idx_list[0])])
    bi_graph['protein'].x = torch.FloatTensor(x_all[len(idx_list[0]):])
    bi_graph['ligand', 'protein'].edge_index = torch.LongTensor(edge_index).t().contiguous()
    bi_graph['ligand', 'protein'].edge_attr = torch.FloatTensor(edge_attr)
    bi_graph = T.ToUndirected()(bi_graph)

    return bi_graph


def generate_graphs(set_dir, process_dir, set_name, cut: int):
    df_file = os.path.join('./data', f'{set_name}.csv')
    df = pd.read_csv(df_file)
    res = {}
    for row in df.itertuples():
        res[row[1]] = row[2]
    # count = 0
    all_list = []
    for item in tqdm(res.keys()):
        score = res[item]
        complex_path = os.path.join(set_dir, item, f'{item}_5A.pkl')

        with open(complex_path, 'rb') as f:
            ligand, pocket = pickle.load(f)

        x1, edge_index1, edge_attr1, pos1 = get_lig_features(ligand)  # 原子的tensor格式的特征，每个(非氢)原子48维度，
        G_lig = Data(x=x1, edge_attr=edge_attr1, edge_index=edge_index1, y=torch.tensor([score]), pos=pos1)
        x2, edge_index2, edge_attr2, pos2 = get_pocket_features(pocket)
        G_poc = Data(x=x2, edge_attr=edge_attr2, edge_index=edge_index2, pos=pos2, code=item)
        bi_graph = get_bigraph(ligand, pocket, cut=cut)
        all_list.append([G_lig, G_poc, bi_graph])

    save_file = os.path.join(process_dir, f'{set_name}_{cut}A.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(all_list, f)


if __name__ == '__main__':
    for set_name in ['train', 'valid', 'test2013', 'test2016']:
        data_root = './data'
        set_dir = os.path.join(data_root, set_name)
        generate_graphs(set_dir, process_dir='./data/processed', set_name=set_name, cut=5)
