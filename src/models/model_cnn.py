import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__)))
import model_base, model_torch_base
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class cnnDataset(Dataset):
    def __init__(self, x, y):
        '''
        cnn用データセットクラス
        '''
        x = np.array(x)
        x = x.reshape((int(x.shape[0]/30), 30, -1))  # (trial, timepoint, features)に変換
        x = x.transpose(0,2,1)  # cnnの入力に合わせ(trial, feature, timepoint)に変換

        y = np.array(y)
        y = y.reshape((int(y.shape[0]/30), 30, -1))

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        out_x = self.x[i]
        out_y = self.y[i]
        return out_x, out_y

class cnn(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.conv = nn.Conv1d(params['in_channels'], params['out_channels'], params['kernel_size'], padding=2)
        self.bn = nn.BatchNorm1d(params['out_channels'])
        self.dropout = nn.Dropout(p = params['p_dropout'])
        self.linear = nn.Linear(params['out_channels'], 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.permute(x, (0,2,1))   # (batch, feature, sequence) -> (b, s, f)
        x = self.dropout(x)
        x = self.linear(x)
        return x

class modeler_cnn(model_torch_base.modeler_torch):
    def __init__(self, params, rand):
        super().__init__(params, rand)
        self.model_class = params['model_class']
        self.dataset_class = cnnDataset

        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.L1Loss()
        # self.loss_fn = RMSELoss()

    def train(self, tr_x, tr_y, es_x, es_y):
        self.params['model_params']['in_channels'] = tr_x.shape[1]

        tr_dataset = cnnDataset(tr_x, tr_y)
        es_dataset = cnnDataset(es_x, es_y)

        super().train(tr_dataset, es_dataset)

    def predict(self, x):
        self.model.eval()

        x = np.array(x)
        x = x.reshape((int(x.shape[0]/30), 30, -1))  # (trial, timepoint, features)に変換
        x = x.transpose(0,2,1)  # cnnの入力サイズ
        x = torch.tensor(x, dtype=torch.float32)

        pred = self.model(x).detach().numpy()
        pred = pred.flatten()
        pred = [float(p) for p in pred]
        return pred

if __name__=='__main__':
    params = {
        'modeltype': 'cnn',
        'rand': 0,
        'use_cv': False,
        'normalize': True,
        'verbose': True,
        'split_by_subject': False,
        'modeler_params': {
            'model_class': cnn,
            'num_epoch': 200,
            'estop_epoch': 20,
            'batch_size': 10,
            'lr': 1e-3,
            'verbose': False,
            'model_params': {'in_channels': None, 'out_channels': 10, 'kernel_size': 5, 'p_dropout':0.3}
        }
    }

    predictor = model_base.vel_prediction(modeler_cnn, params)
    predictor.main()
