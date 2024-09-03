from torch_geometric.data import DataLoader
import os
from dataset import MyDataset
from utils.utils import *
from model.TdmNet import Tdm_Net


def val(model, val_loader, device):
    p_list = []
    y_list = []
    model.eval()
    for step, data in enumerate(val_loader):
        with torch.no_grad():
            data = set_device(data, device)
            pre = model(data)
            p_list.extend(pre.detach().cpu().tolist())
            y_list.extend(data[0].y.detach().cpu().tolist())
    return metrics_result(targets=y_list, predicts=p_list)


if __name__ == '__main__':
    device = torch.device("cuda")
    print("loading dataset...")

    model = Tdm_Net(256).to(device)
    model.load_state_dict(torch.load(
        "./output/20240902_142533_TdmNet/model/epoch-512, val_mae-0.9029, val_rmse-1.1749, val_pr-0.7755.pt"))

    data_root = './data/processed'

    test2013_path = os.path.join(data_root, 'test2013_5A.pkl')
    test2016_path = os.path.join(data_root, 'test2016_5A.pkl')

    test2013_set = MyDataset(file_path=test2013_path)
    test2016_set = MyDataset(file_path=test2016_path)

    test2013_loader = DataLoader(test2013_set, batch_size=128, shuffle=False)
    test2016_loader = DataLoader(test2016_set, batch_size=128, shuffle=False)

    val_mae, val_rmse, val_pr = val(model, test2013_loader, device)
    print(val_mae)
    print(val_rmse)
    print(val_pr)


    val_mae, val_rmse, val_pr = val(model, test2016_loader, device)
    print(val_mae)
    print(val_rmse)
    print(val_pr)