import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__)))
import model_base
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class customDataset(Dataset):
    def __init__(self, x, y):
        x = np.array(x)
        x = x.reshape((int(x.shape[0]/30), 30, -1))  # (trial, timepoint, features)に変換

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

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, y):
        return ((pred - y)**2).mean() **0.5

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
        self.rnn = nn.LSTM(params['input_size'], params['hidden_size'])
        self.dropout = nn.Dropout(p=params['p_dropout'])
        self.fc = nn.Linear(params['hidden_size'], 1)

    def forward(self, x):
        x, h = self.rnn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class modeler_rnn(model_base.modeler_base):
    def __init__(self, params, rand):
        self.params = params
        self.rand = rand

        self.model = None
        self.optimizer = None
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.L1Loss()
        # self.loss_fn = RMSELoss()
    
    def train_loop(self, dataloader):
        self.model.train()
        for batch, (x, y) in enumerate(dataloader):
            pred = self.model(x)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def test_loop(self, dataloader):
        self.model.eval()
        truth = list()
        pred = list()
        
        for x, y in dataloader:
            truth.extend(y.detach().numpy().flatten())
            pred.extend(self.model(x).detach().numpy().flatten())
        
        truth = np.array(truth)
        pred = np.array(pred)
        rmse = model_base.rmse(pred, truth)
        # print(f'rmse: {rmse}')
        return rmse

    def train(self, tr_x, tr_y, es_x, es_y):
        self.params['rnn_params']['input_size'] = tr_x.shape[1]

        # self.model = simplernn(self.params['rnn_params'])
        # self.model = gru(self.params['rnn_params'])
        self.model = lstm(self.params['rnn_params'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])

        # データローダー
        train_dataset = customDataset(tr_x, tr_y)
        estop_dataset = customDataset(es_x, es_y)

        train_dataloader = DataLoader(train_dataset, batch_size=self.params['batch_size'], shuffle=True)
        estop_dataloader = DataLoader(estop_dataset, batch_size=self.params['batch_size'], shuffle=True)

        # 記録用ndarray
        self.log = np.zeros((self.params['num_epoch'], 2))
        ep = max(math.ceil(self.params['num_epoch']/10), 1)

        for epoch in range(self.params['num_epoch']):
            self.train_loop(train_dataloader)

            self.log[epoch, 0] = self.test_loop(train_dataloader)
            self.log[epoch, 1] = self.test_loop(estop_dataloader)

            if epoch%ep == 0:
                print(f'estop rmse: {self.log[epoch, 1]} [{epoch}/{self.params["num_epoch"]}]')
        
        # 学習曲線描画
        plt.figure(figsize=(5,3))
        plt.xlabel('epochs')
        plt.ylabel('rmse')
        plt.plot(self.log)
        plt.show()

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
            'rnn_params': {'input_size': None, 'hidden_size': 50, 'p_dropout': 0.7}
        }
    }

    predictor = model_base.vel_prediction(modeler_rnn, params)
    predictor.main()
