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

class cnnDataset(Dataset):
    def __init__(self, x, y, loss_seq):
        x = np.array(x)
        x = x.reshape((int(x.shape[0]/30), 30, -1))  # (trial, timepoint, features)に変換
        x = x.transpose(0,2,1)  # cnnの入力に合わせ(trial, feature, timepoint)に変換

        y = np.array(y)
        y = y.reshape((int(y.shape[0]/30), 30, -1))

        # cnnのkernel_sizeに応じて出力のシーケンス長が変化するのでyを合わせる
        out_seq = 30 - loss_seq    # 出力シーケンス長
        y = y[:,-out_seq:,:]

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

class cnn(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.conv = nn.Conv1d(params['in_channels'], params['out_channels'], params['kernel_size'])
        self.dropout = nn.Dropout(p = params['p_dropout'])
        self.linear = nn.Linear(params['out_channels'], 1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.permute(x, (0,2,1))   # (batch, feature, sequence) -> (b, s, f)
        x = self.dropout(x)
        x = self.linear(x)
        return x

class modeler_cnn(model_base.modeler_base):
    def __init__(self, params, rand):
        self.params = params
        self.rand = rand

        self.model = None
        self.optimizer = None
        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.L1Loss()
        # self.loss_fn = RMSELoss()

        self.loss_seq = self.params['cnn_params']['kernel_size'] - 1    # cnnによるシーケンスの縮小幅
    
    def train_loop(self, dataloader):
        self.model.train()
        for x, y in dataloader:
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

    def train(self, tr_x, tr_y, es_x, es_y):
        self.params['cnn_params']['in_channels'] = tr_x.shape[1]

        self.model = cnn(self.params['cnn_params'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])

        # データローダー
        train_dataset = cnnDataset(tr_x, tr_y, self.loss_seq)
        estop_dataset = cnnDataset(es_x, es_y, self.loss_seq)

        train_dataloader = DataLoader(train_dataset, batch_size=self.params['batch_size'], shuffle=True)
        estop_dataloader = DataLoader(estop_dataset, batch_size=self.params['batch_size'], shuffle=True)

        # 記録用ndarray
        self.log = np.zeros((self.params['num_epoch'], 2))
        ep = max(math.ceil(self.params['num_epoch']/10), 1)

        for epoch in range(self.params['num_epoch']):
            self.train_loop(train_dataloader)

            self.log[epoch, 0] = self.test_loop(train_dataloader)
            self.log[epoch, 1] = self.test_loop(estop_dataloader)

            if epoch%(ep) == 0:
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
        x = x.transpose(0,2,1)  # cnnの入力サイズ
        print(x.shape)
        x = torch.tensor(x, dtype=torch.float32)

        pred = self.model(x).detach().numpy()

        # 縮小幅の復元
        pred = pred.reshape(pred.shape[0], pred.shape[1])
        pred = np.array([np.concatenate([np.array([p.mean()]*self.loss_seq), p]) for p in pred])    # ！縮小幅を埋める
        pred = pred.flatten()
        pred = [float(p) for p in pred]
        return pred
