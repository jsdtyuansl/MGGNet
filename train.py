import warnings

warnings.filterwarnings('ignore')
from torch.optim.lr_scheduler import StepLR
import argparse
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from log.train_logger import TrainLogger
from utils.utils import *
from dataset import MyDataset
from model.MGGNet import MGGNet

seed = 42
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(train_loader, val_loader, test_set):
    print('start training...')
    args = parse_arguments()
    logger = TrainLogger(args, create=True)

    model = MGGNet(256).to(device)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-6)

    train_loss_all = []
    best_mae = float('inf')
    best_rmse = float('inf')
    patience = 0
    best_path = ''
    model.train()

    for epoch in range(500):
        loss_epoch = 0
        train_num = 0
        for step, data in enumerate(train_loader):
            data = set_device(data, device)
            out = model(data)
            loss = loss_func(out, data[0].y)
            loss_epoch += loss.item() * data[0].y.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_num += data[0].y.size(0)

        train_loss_all.append(loss_epoch / train_num)
        val_mae, val_rmse, val_pr = val(model, val_loader, device)

        if val_rmse < best_rmse:
            print('---save model---')
            patience = 0
            title = "epoch-%d, val_mae-%.4f, val_rmse-%.4f, val_pr-%.4f" % (epoch, val_mae, val_rmse, val_pr)
            best_path = save_model_dict(model, logger.get_model_dir(), title)
            best_rmse = val_rmse
            logger.info(title)
        else:
            patience += 1
            if patience >= 100:
                logger.info(f"Early stopping after {epoch}")
                info = "best_rmse-%.4f" % best_rmse
                logger.info(info)
                break

    # final test
    test2013_mae, test2013_rmse, test2013_pr = test(test_set[0], best_path)
    test2016_mae, test2016_rmse, test2016_pr = test(test_set[1], best_path)

    info1 = "test2013_mae-%.4f, test2013_rmse-%.4f, test2013_pr-%.4f" % (test2013_mae, test2013_rmse, test2013_pr)
    logger.info(info1)
    info2 = "test2016_mae-%.4f, test2016_rmse-%.4f, test2016_pr-%.4f" % (test2016_mae, test2016_rmse, test2016_pr)
    logger.info(info2)


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
    model.train()
    return metrics_result(targets=y_list, predicts=p_list)


def test(test_set, model_file):
    p_list = []
    y_list = []
    m_state_dict = torch.load(model_file)
    best_model = MGGNet(256).to(device)
    best_model.load_state_dict(m_state_dict)
    test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=False)
    best_model.eval()
    for step, data in enumerate(test_loader):
        with torch.no_grad():
            data = set_device(data, device)
            pre = best_model(data)
            p_list.extend(pre.detach().cpu().tolist())
            y_list.extend(data[0].y.detach().cpu().tolist())
    return metrics_result(targets=y_list, predicts=p_list)


def parse_arguments():
    parser = argparse.ArgumentParser(description='TdmNet')
    parser.add_argument('--model', default="TdmNet", help='Where is the raw data')
    parser.add_argument('--data_root', default="./data/processed", help='Where is the raw data')
    parser.add_argument('--save_dir', default="./output", help="Where to save the result")
    parser.add_argument('--save_model', default=True, help='Save the model or not')
    parser.add_argument('--epochs', default=600, help="Train epochs")
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--fusion', choices=['Sum', 'Dot', 'Concat', 'Average'], default='Concat',
                        help="Select Fusion Method")
    parser.add_argument('--lr', default=0.0005, help="Learning Rate")
    parser.add_argument('--wd', default=5e-5, help="weight_decay")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    data_root = args.data_root
    save_path = args.save_dir
    save_model = args.save_model
    batch_size = args.batch_size
    epochs = args.epochs


    for repeat in range(1):
        print("loading dataset...")
        train_path = os.path.join(data_root, 'train_5A.pkl')
        valid_path = os.path.join(data_root, 'valid_5A.pkl')
        test2013_path = os.path.join(data_root, 'test2013_5A.pkl')
        test2016_path = os.path.join(data_root, 'test2016_5A.pkl')

        train_set = MyDataset("file", train_path)
        valid_set = MyDataset("file", valid_path)
        test2013_set = MyDataset("file", test2013_path)
        test2016_set = MyDataset("file", test2016_path)
        test_set = [test2013_set, test2016_set]

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=True)
        train(train_loader, valid_loader, test_set)
