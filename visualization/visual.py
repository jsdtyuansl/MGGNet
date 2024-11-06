import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import torch
from torch_geometric.data import Batch
import pandas as pd
from matplotlib.colors import ListedColormap
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor

rdDepictor.SetPreferCoordGen(True)
from IPython.display import SVG
import cairosvg
import cv2
import matplotlib.cm as cm
from tqdm import tqdm
from dataset import MyDataset
from torch_geometric.data import DataLoader
from dataset import *
from model.MGGNet import MGGNet


class GradAAM():
    def __init__(self, model, module):
        self.model = model
        module.register_forward_hook(
            self.save_hook)
        self.target_feat = None

    def save_hook(self, md, fin, fout):  # 初始化一个变量 target_feat，用于存储 module 的输出特征
        self.target_feat = fout

    def __call__(self, data):
        self.model.eval()

        output = self.model(data)
        feat = self.target_feat
        grad = torch.autograd.grad(output, feat)[0]
        channel_weight = torch.mean(grad, dim=0, keepdim=True)  # 计算梯度的均值，保留维度
        channel_weight = normalize(channel_weight)
        weighted_feat = feat * channel_weight
        cam = torch.sum(weighted_feat,
                        dim=-1).detach().cpu().numpy()  # 计算加权特征的总和
        cam = normalize(cam)
        return output.detach().cpu().numpy(), cam


def clourMol(mol, highlightAtoms_p=None, highlightAtomColors_p=None, highlightBonds_p=None, highlightBondColors_p=None,
             r=None):
    d2d = rdMolDraw2D.MolDraw2DSVG(400, 400)
    op = d2d.drawOptions()
    op.dotsPerAngstrom = 40
    op.useBWAtomPalette()
    mc = rdMolDraw2D.PrepareMolForDrawing(mol)
    d2d.DrawMolecule(mc, legend='', highlightAtoms=highlightAtoms_p, highlightAtomColors=highlightAtomColors_p,
                     highlightBonds=highlightBonds_p, highlightBondColors=highlightBondColors_p,
                     highlightAtomRadii=r)
    d2d.FinishDrawing()
    svg = SVG(d2d.GetDrawingText())
    res = cairosvg.svg2png(svg.data, dpi=600, output_width=2400, output_height=2400)
    nparr = np.frombuffer(res, dtype=np.uint8)
    segment_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return segment_data


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)


def set_device(data, device):
    data_device = []
    for g in data:
        data_device.append(g.to(device))
    return data_device


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MGGNet(256).to(device)
    ckpt = "./pretrain/epoch-334, val_mae-0.8940, val_rmse-1.1674, val_pr-0.7769.pt"
    model.load_state_dict(torch.load(ckpt))
    model.eval()

    gradcam = GradAAM(model, model.ligandnet.conv_3)

    data_root = '../data/processed'
    draw_path = os.path.join(data_root, 'draw_set_5A.pkl')
    draw_set = MyDataset("file", draw_path)
    loader = DataLoader(dataset=draw_set, batch_size=1, shuffle=False)

    test_list = os.listdir("../data/draw_set")

    for n, data in enumerate(loader):
        data = set_device(data, device)
        filename = os.path.join(test_list[n], test_list[n] + "_ligand.pdb")
        filename = os.path.join("../data/draw_set", filename)
        _, att = gradcam(data)

        mol = Chem.MolFromPDBFile(filename)

        for idx in range(mol.GetNumAtoms()):
            mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(idx))

        # nums = mol.GetNumAtoms()
        # assert att.shape != nums, "原子数不该不等"

        smi = Chem.MolToSmiles(mol)
        mol_smi = Chem.MolFromSmiles(smi)

        att_list = [att[int(mol_smi.GetAtomWithIdx(idx).GetProp('molAtomMapNumber'))] for idx in
                    range(mol_smi.GetNumAtoms())]
        atom_att = np.array(att_list)
        for atom in mol_smi.GetAtoms():
            atom.ClearProp('molAtomMapNumber')

        id = []
        r = {}
        threshold = np.mean(atom_att)
        print(threshold)
        for i in range(len(atom_att)):
            if atom_att[i] > 0.25:
                id.append(i)

        r = dict([(idx, 0.25) for idx in range(len(atom_att))])
        img = clourMol(mol_smi, highlightAtoms_p=id, r=r)
        cv2.imwrite(os.path.join('./result', f'{test_list[n]}.png'), img)
