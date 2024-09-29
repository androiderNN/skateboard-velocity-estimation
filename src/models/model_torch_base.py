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
        '''
        汎用データセットクラス
        (trial, timepoint, features)を保持する'''
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

class modeler_torch(model_base.modeler_base):
    def __init__(self, params, rand):
        self.params = params
        self.rand = rand

        self.model_class = None
        self.dataset_class = None
        self.model = None
        self.optimizer = None
        self.loss_fn = None
    
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
        return rmse

    def train(self, train_dataset, estop_dataset):
        '''
        trainとestopのdatasetを入力すると学習とログ出力を行う'''
        self.model = self.model_class(self.params['model_params'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])

        # データローダー
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
        if self.params['verbose']:
            plt.figure(figsize=(5,3))
            plt.xlabel('epochs')
            plt.ylabel('rmse')
            plt.plot(self.log)
            plt.show()
