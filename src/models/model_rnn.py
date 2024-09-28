import os, sys, pprint
import numpy as np
import pandas as pd
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

class gru(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.rnn = nn.GRU(params['input_size'], params['hidden_size'])
        self.fc = nn.Linear(params['hidden_size'], 1)

    def forward(self, x):
        x, h = self.rnn(x, )
        x = self.fc(x)
        return x

class modeler_rnn(model_base.modeler_base):
    def __init__(self, params):
        self.params = params

        self.model = None
        self.optimizer = None
        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.L1Loss()
    
    def train_loop(self, dataloader):
        self.model.train()
        for batch, (x, y) in enumerate(dataloader):
            pred = self.model(x)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch%100==0:
                loss = loss.item()**0.5
                current = batch
                # print(f'loss: {loss} [{current}]')

    def test_loop(self, dataloader):
        self.model.eval()
        size = len(dataloader.dataset)
        truth = list()
        pred = list()
        
        for x, y in dataloader:
            truth.extend(y.detach().numpy().flatten())
            pred.extend(self.model(x).detach().numpy().flatten())
        
        truth = np.array(truth)
        pred = np.array(pred)
        rmse = model_base.rmse(pred, truth)
        print(f'rmse: {rmse}')

    def train(self, tr_x, tr_y, es_x, es_y):
        # print(tr_x.iloc[:,-5:])
        # tr_x.fillna(0, inplace=True)
        self.params['gru_params']['input_size'] = tr_x.shape[1]

        self.model = gru(self.params['gru_params'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params['lr'])

        train_dataset = customDataset(tr_x, tr_y)
        estop_dataset = customDataset(es_x, es_y)

        train_dataloader = DataLoader(train_dataset, batch_size=self.params['batch_size'], shuffle=True)
        estop_dataloader = DataLoader(estop_dataset, batch_size=self.params['batch_size'], shuffle=True)

        for epoch in range(self.params['num_epoch']):
            self.train_loop(train_dataloader)

            if epoch%10 == 0:
                self.test_loop(estop_dataloader)

    def predict(self, x):
        self.model.eval()

        x = np.array(x)
        x = x.reshape((int(x.shape[0]/30), 30, -1))  # (trial, timepoint, features)に変換
        x = torch.tensor(x, dtype=torch.float32)

        pred = self.model(x).detach().numpy().flatten()
        return pred

