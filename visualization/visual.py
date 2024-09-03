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
from model.TdmNet import Tdm_Net


class GradAAM():
    def __init__(self, model, module):
        self.model = model
        module.register_forward_hook(
            self.save_hook)  # 注册一个名为 save_hook 的函数作为 module 的前向传播钩子。这意味着在 module 的前向传播完成后，会自动调用 save_hook 函数
        self.target_feat = None

    def save_hook(self, md, fin, fout):  # 初始化一个变量 target_feat，用于存储 module 的输出特征
        self.target_feat = fout  # 在钩子函数中，将 module 的输出赋值给 target_feat

    def __call__(self, data):
        self.model.eval()

        output = self.model(data).view(-1)
        grad = torch.autograd.grad(output, self.target_feat)[0]
        channel_weight = torch.mean(grad, dim=0, keepdim=True)  # 计算梯度的均值，保留维度
        channel_weight = normalize(channel_weight)
        weighted_feat = self.target_feat * channel_weight
        cam = torch.sum(weighted_feat,
                        dim=-1).detach().cpu().numpy()  # 计算加权特征的总和，然后将其从计算图中分离（detach），移动到 CPU 并转换为 NumPy 数组
        cam = normalize(cam)

        return output.detach().cpu().numpy(), cam


def clourMol(mol, highlightAtoms_p=None, highlightAtomColors_p=None, highlightBonds_p=None, highlightBondColors_p=None,
             sz=[400, 400], radii=None):
    d2d = rdMolDraw2D.MolDraw2DSVG(sz[0], sz[1])
    op = d2d.drawOptions()
    op.dotsPerAngstrom = 40
    op.useBWAtomPalette()
    mc = rdMolDraw2D.PrepareMolForDrawing(mol)
    d2d.DrawMolecule(mc, legend='', highlightAtoms=highlightAtoms_p, highlightAtomColors=highlightAtomColors_p,
                     highlightBonds=highlightBonds_p, highlightBondColors=highlightBondColors_p,
                     highlightAtomRadii=radii)
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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Tdm_Net().to(device)
    ckpt = "./pretrain/epoch-378, val_mae-0.8993, val_rmse-1.2239.pt"
    model.load_state_dict(torch.load(ckpt))
    model.eval()

    for name, module in model.ligandnet._modules.items():
        if (name == "conv_3"):
            gradcam = GradAAM(model, module=module)
            break;

    bottom = cm.get_cmap('Blues_r', 256)
    top = cm.get_cmap('Oranges', 256)
    newcolors = np.vstack([bottom(np.linspace(0.35, 0.85, 128)), top(np.linspace(0.15, 0.65, 128))])
    newcmp = ListedColormap(newcolors, name='OrangeBlue')

    test_set = MyDataset("file", "../data/processed", "test1")
    dataloader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    test_list = os.listdir("../data/test1_set")

    for n, data in enumerate(dataloader):
        data = set_device(data, device)
        filename = os.path.join(test_list[n], test_list[n] + "_ligand.pdb")
        filename = os.path.join("../data/test1_set", filename)
        _, atom_att = gradcam(data)
        mol = Chem.MolFromPDBFile(filename)
        smi = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smi)

        atom_color = dict([(idx, newcmp(atom_att[idx])[:3]) for idx in range(len(atom_att))])
        radii = dict([(idx, 0.2) for idx in range(len(atom_att))])
        img = clourMol(mol, highlightAtoms_p=range(len(atom_att)), highlightAtomColors_p=atom_color, radii=radii)
        cv2.imwrite(os.path.join('./result', f'{test_list[n]}.png'), img)


if __name__ == '__main__':
    main()
