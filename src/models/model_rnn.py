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

class simplernn(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.rnn = nn.RNN(params['input_size'], params['hidden_size'])
        self.dropout = nn.Dropout(p=params['p_dropout'])
        self.fc = nn.Linear(params['hidden_size'], 1)

    def forward(self, x):
        x, h = self.rnn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class gru(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.rnn = nn.GRU(params['input_size'], params['hidden_size'])
        self.dropout = nn.Dropout(p=params['p_dropout'])
        self.fc = nn.Linear(params['hidden_size'], 1)

    def forward(self, x):
        x, h = self.rnn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class lstm(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.rnn = nn.LSTM(params['input_size'], params['hidden_size'], num_layers=params['num_layers'], dropout=params['dropout'])
        self.dropout = nn.Dropout(p=params['p_dropout'])
        self.fc = nn.Linear(params['hidden_size'], 1)
        # self.bn = nn.BatchNorm2d(30)

    def forward(self, x):
        x, h = self.rnn(x)

        # shape = x.shape
        # x = x.reshape((shape[0], shape[1], shape[2], 1))   # batch norm 2dに入力するためreshape
        # x = self.bn(x)
        # x = x.reshape(shape)

        x = self.dropout(x)
        x = self.fc(x)
        return x

class lstm2(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.rnn = nn.LSTM(params['input_size'], params['hidden_size'], num_layers=params['num_layers'], dropout=params['dropout'])
        self.linear1 = nn.Linear(params['hidden_size'], params['hidden_size'])
        self.linear2 = nn.Linear(params['hidden_size'], 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=params['p_dropout'])
        # self.bn1 = nn.BatchNorm2d(30)
        # self.bn2 = nn.BatchNorm2d(30)

    def forward(self, x):
        x, h = self.rnn(x)

        # shape = x.shape
        # x = x.reshape((shape[0], shape[1], shape[2], 1))   # batch norm 2dに入力するためreshape
        # x = self.bn1(x)
        # x = x.reshape(shape)

        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)

        # x = x.reshape((shape[0], shape[1], shape[2], 1))   # batch norm 2dに入力するためreshape
        # x = self.bn2(x)
        # x = x.reshape(shape)

        x = self.dropout(x)
        x = self.linear2(x)
        return x

class modeler_rnn(model_torch_base.modeler_torch):
    def __init__(self, params, rand):
        super().__init__(params, rand)
        # self.model_class = rnn
        # self.model_class gru
        self.model_class = params['model_class']
        self.dataset_class = model_torch_base.dataset_df

        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.L1Loss()
        # self.loss_fn = model_torch_base.RMSELoss()
    
    def train(self, tr_x, tr_y, es_x, es_y):
        self.params['model_params']['input_size'] = tr_x.shape[1]

        tr_dataset = self.dataset_class(tr_x, tr_y)
        es_dataset = self.dataset_class(es_x, es_y)

        super().train(tr_dataset, es_dataset)

    def predict(self, x):
        self.model.eval()

        x = np.array(x)
        x = x.reshape((int(x.shape[0]/30), 30, -1))  # (trial, timepoint, features)に変換
        x = torch.tensor(x, dtype=torch.float32)

        pred = self.model(x).detach().numpy().flatten()
        pred = [float(p) for p in pred]
        return pred

if __name__=='__main__':
    params = {
        'modeltype': 'lstm',
        'rand': 0,
        'use_cv': False,
        'normalize': True,
        'smoothing': 'ma',
        'verbose': True,
        'split_by_subject': False,
        'modeler_params': {
            'model_class': lstm,
            'num_epoch': 200,
            'estop_epoch': 20,
            'batch_size': 10,
            'lr': 1e-3,
            'verbose': False,
            'model_params': {'input_size': None, 'hidden_size': 50, 'p_dropout': 0.7, 'num_layers': 2, 'dropout':0.7}
        }
    }

    print(params['modeler_params']['model_params'])

    predictor = model_base.vel_prediction(modeler_rnn, params)
    predictor.main()
