import os
import numpy as np
import torch
import random
import pandas as pd

from dataset import MyDataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model.MGGNet import Tdm_Net
from torch_geometric.data import Batch, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    model = Tdm_Net(256).to(device)
    ckpt = "./pretrain/epoch-526, val_mae-0.8912, val_rmse-1.1692, val_pr-0.7750.pt"
    model.load_state_dict(torch.load(ckpt))

    data_root = '../data/processed'

    train_path = os.path.join(data_root, 'train_5A.pkl')
    test2016_path = os.path.join(data_root, 'test2016_5A.pkl')

    train_set = MyDataset("file", train_path)
    test2016_set = MyDataset("file", test2016_path)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test2016_loader = DataLoader(test2016_set, batch_size=1, shuffle=False)


    def set_device(data, device):
        data_device = []
        for g in data:
            data_device.append(g.to(device))
        return data_device


    def normalize(x):
        return (x - x.min()) / (x.max() - x.min())


    # 创建一个 t-SNE 模型并将数据降维到 2 维
    def tsne(title: str, feats, c, cmap=None, vmin=None, vmax=None):
        tsne = TSNE(n_components=2, perplexity=30,early_exaggeration=20, random_state=42)
        tsne_result = tsne.fit_transform(feats)
        plt.figure(figsize=(10, 6))
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], marker='o', s=70, c=c, cmap=cmap, vmin=vmin,
                    vmax=vmax)

        # 显示坐标轴边框
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.savefig(title, dpi=600)
        plt.show()


    class UseHook():
        def __init__(self, model, module):
            self.model = model
            module.register_forward_hook(self.save_hook)

        def save_hook(self, md, fin, fout):
            self.target_feat = fout

        def __call__(self, data):
            self.model.eval()
            output = self.model(data)
            return self.target_feat[0].detach().cpu().numpy()


    class UseHook2():
        def __init__(self, model, module):
            self.model = model
            module.register_forward_hook(self.save_hook)

        def save_hook(self, md, fin, fout):
            self.target_feat = fout

        def __call__(self, data):
            self.model.eval()
            output = self.model(data)
            return self.target_feat['ligand'].detach().cpu().numpy(), self.target_feat['protein'].detach().cpu().numpy()


    # print(model.complexnet)
    hook1 = UseHook(model, model.out.predict[6])
    hook2 = UseHook(model, model.ligandnet)
    hook3 = UseHook(model, model.proteinnet)


    random.seed(42)
    random_index = list(range(0, 11899))
    random.shuffle(random_index)
    index_list = random_index[:256]
    feats_hook1 = []
    feats_hook2 = []
    feats_hook3 = []

    feats_hook5 = []
    affinity = []

    data_list = train_set.get_by_idx(index_list)
    sample_data = MyDataset("list", data_list)
    loader = DataLoader(dataset=sample_data, batch_size=1)

    for data in loader:
        data = set_device(data, device)
        label = data[0].y.cpu().numpy()
        feats_hook1.append(hook1(data))
        affinity.append(label)
    arr1 = np.array(feats_hook1)

    print(arr1)
    affinity = np.array(affinity).reshape(-1)
    affinity = normalize(affinity)

    tsne("./result/tsne_epoch526.png", arr1, c=affinity, cmap='coolwarm', vmin=0, vmax=1)

    # tsne = TSNE(n_components=2, perplexity=30, early_exaggeration=20, random_state=42)
    # tsne_result = tsne.fit_transform(feats)
    # plt.figure(figsize=(10, 6))
    # plt.scatter(tsne_result[:285, 0], tsne_result[:285, 1], marker='o', s=70, c="cyan")
    # plt.scatter(tsne_result[285:, 0], tsne_result[285:, 1], marker='o', s=70, c="yellow")
    #
    # # 显示坐标轴边框
    # plt.gca().xaxis.set_ticks_position('none')
    # plt.gca().yaxis.set_ticks_position('none')
    # plt.savefig("./result/epoch526_two_test.png", dpi=600)
    # plt.show()






    # for data in test2016_loader:
    #     data = set_device(data, device)
    #     label = data[0].y.cpu().numpy()
    #     feats_hook2.append(hook2(data))
    #     feats_hook3.append(hook3(data))
    #     affinity.append(label)
    #
    # arr1 = np.array(feats_hook2)
    # arr2 = np.array(feats_hook3)
    # # print(np.vstack((arr1,arr2)).shape)
    #
    # feats = np.vstack((arr1, arr2))
    #
    # print(feats.shape)
    #
    # affinity = np.array(affinity).reshape(-1)
    # affinity = normalize(affinity)
    #
    # tsne("./result/tsne_epoch526.png", feats, color='coolwarm', cmap=affinity, vmin=0, vmax=1)
    #
    # tsne = TSNE(n_components=2, perplexity=30, early_exaggeration=20, random_state=42)
    # tsne_result = tsne.fit_transform(feats)
    # plt.figure(figsize=(10, 6))
    # plt.scatter(tsne_result[:285, 0], tsne_result[:285, 1], marker='o', s=70, c="cyan")
    # plt.scatter(tsne_result[285:, 0], tsne_result[285:, 1], marker='o', s=70, c="yellow")
    #
    # # 显示坐标轴边框
    # plt.gca().xaxis.set_ticks_position('none')
    # plt.gca().yaxis.set_ticks_position('none')
    # plt.savefig("./result/epoch526_two_test.png", dpi=600)
    # plt.show()












    hook2 = UseHook(model, model.ligandnet.attention)
    hook3 = UseHook(model, model.proteinnet.attention)
    hook5 = UseHook2(model, model.complexnet.conv_3)

    for data in test2016_loader:
        data = set_device(data, device)
        feats_hook2.append(hook2(data))
        feats_hook3.append(hook3(data))
        feats_hook5.append(hook5(data)[0])
        feats_hook5.append(hook5(data)[1])
        break
    print(feats_hook5[0].shape)
    print(feats_hook5[1].shape)
    arr1 = np.array(feats_hook2).squeeze(0)
    print(arr1)
    arr2 = np.array(feats_hook3).squeeze(0)
    print(arr2)
    arr3 = np.array(feats_hook5[0])
    arr4 = np.array(feats_hook5[1])

    print(arr1.shape)
    print(arr2.shape)
    print(arr3.shape)

    feats_complex = np.concatenate((arr3, arr4), axis=0)
    print(feats_complex.shape)

    feats = np.vstack((arr1, feats_complex))

    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(feats)

    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_result[:26, 0], tsne_result[:26, 1], marker='o', s=70, c="cyan")
    plt.scatter(tsne_result[26:, 0], tsne_result[26:, 1], marker='o', s=70, c="green")

    # 显示坐标轴边框
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.savefig("./result/epoch526_two_lc.png", dpi=600)
    plt.show()
