import torch
import os
from sklearn import metrics
from scipy.stats import pearsonr

def set_device(data, device):
    data_device = []
    for g in data:
        data_device.append(g.to(device))
    return data_device

def metrics_result(targets, predicts):
    mae = metrics.mean_absolute_error(y_true=targets, y_pred=predicts)
    rmse = metrics.mean_squared_error(y_true=targets, y_pred=predicts, squared=False)
    r = pearsonr(targets, predicts)[0]
    return [mae, rmse, r]

def save_model_dict(model, save_path, title):
    model_path = os.path.join(save_path, title + '.pt')
    torch.save(model.state_dict(), model_path)
    return model_path
    # print("model has been saved to %s." % (model_path))


def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))
