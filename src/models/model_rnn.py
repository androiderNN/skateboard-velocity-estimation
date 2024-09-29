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

    def forward(self, x):
        x, h = self.rnn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class modeler_rnn(model_torch_base.modeler_torch):
    def __init__(self, params, rand):
        super().__init__(params, rand)
        # self.model_class = rnn
        # self.model_class gru
        self.model_class = lstm
        self.dataset_class = model_torch_base.customDataset

        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.L1Loss()
        # self.loss_fn = model_torch_base.RMSELoss()
    
    def train(self, tr_x, tr_y, es_x, es_y):
        self.params['model_params']['input_size'] = tr_x.shape[1]

        tr_dataset = model_torch_base.customDataset(tr_x, tr_y)
        es_dataset = model_torch_base.customDataset(es_x, es_y)

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
        'modeltype': 'rnn',
        'rand': 0,
        'use_cv': False,
        'normalize': True,
        'verbose': True,
        'split_by_subject': False,
        'modeler_params': {
            'num_epoch': 50,
            'batch_size': 10,
            'lr': 1e-3,
            'verbose': False,
            'model_params': {'input_size': None, 'hidden_size': 50, 'p_dropout': 0.7, 'num_layers': 1, 'dropout':0}
        }
    }

    predictor = model_base.vel_prediction(modeler_rnn, params)
    predictor.main()
