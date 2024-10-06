import os, sys, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from features import iemg
from models import model_torch_base, model_cnn

class dataset_cnnf(Dataset):
    def __init__(self, myodata, target):
        '''
        myodata: ndarray (batch, channel, timepoint)'''
        super().__init__()
        # self.myodata = torch.tensor(myodata.reshape(-1, 16, 1000))    # (batch, feature, seq)
        self.myodata = torch.tensor(myodata, dtype=torch.float32)
        # self.target = torch.tensor(target[:, np.newaxis], dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)
    
    def __len__(self):
        return self.myodata.shape[0]

    def __getitem__(self, idx):
        return self.myodata[idx], self.target[idx]

def myo_processor(raw_data, lowcut):
    myo = [raw_data[str(s).zfill(4)][0,0][0] for s in range(1,5)]   # 筋電位データのみ抽出
    myo = [iemg.iemg_core(m, lowcut) for m in myo]  # フィルタリング
    num_trials = [m.shape[0] for m in myo]

    myo = np.array([t for m in myo for t in m]) # (12**, 16, 1000)
    return myo, num_trials

def initvel_processor(raw_data):
    initvel = np.array([t[0,0] for s in range(1,5) for t in raw_data[str(s).zfill(4)][0,0][1]]) # 各trialのxの初速
    return initvel

def finvel_processor(raw_data):
    initvel = np.array([t[0,29] for s in range(1,5) for t in raw_data[str(s).zfill(4)][0,0][1]]) # 各trialのxの初速
    return initvel

# def xmin_processor(raw_data):
#     xmin = np.array([np.argmin(t[0,:]) for s in range(1,5) for t in raw_data[str(s).zfill(4)][0,0][1]]) # 各trialのxの初速
#     return xmin

def vel_processor(raw_data):
    vel = np.array([t[0,:] for s in range(1,5) for t in raw_data[str(s).zfill(4)][0,0][1]]) # 各trialのx速度
    return vel

def xzero_30features(raw_data):
    xmin = np.array([t[0,:] for s in range(1,5) for t in raw_data[str(s).zfill(4)][0,0][1]]) # 各trialのxの初速
    y = np.zeros((xmin.shape[0], 30))
    for i, x in enumerate(xmin):
        y[i][x>=0] = 1
    
    y = y.reshape((y.shape[0], -1))
    return y

class cnn1(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.conv1 = nn.Conv1d(params['conv1_in'], params['conv1_out'], params['conv1_ksize'], params['conv1_stride'])
        self.conv2 = nn.Conv1d(params['conv1_out'], params['conv2_out'], params['conv2_ksize'], params['conv2_stride'])
        self.conv3 = nn.Conv1d(params['conv2_out'], params['conv3_out'], params['conv3_ksize'], params['conv3_stride'])
        self.conv4 = nn.Conv1d(params['conv3_out'], params['conv4_out'], params['conv4_ksize'], params['conv4_stride'])
        self.pool1 = nn.MaxPool1d(kernel_size=params['pool1_ksize'])
        self.pool2 = nn.MaxPool1d(kernel_size=params['pool2_ksize'])
        self.pool3 = nn.MaxPool1d(kernel_size=params['pool3_ksize'])
        self.linear1 = nn.Linear(params['conv4_out'], params['conv4_out'])
        self.linear2 = nn.Linear(params['conv4_out'], 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=params['p_dropout'])
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class cnn1_(nn.Module):
    def __init__(self, params):
        '''
        {
        'conv1_in': 16,
        'conv1_out': 128,
        'conv1_ksize': 10,
        'conv1_stride': 3,
        'conv2_out': 256,
        'conv2_ksize': 5,
        'conv2_stride': 2,
        'conv3_out': 512,
        'conv3_ksize': 3,
        'conv3_stride': 1,
        'conv4_out': 512,
        'conv4_ksize': 3,
        'conv4_stride': 1,
        'pool1_ksize': 4,
        'pool2_ksize': 4,
        'pool3_ksize': 4,
        'p_dropout': 0.5,
        }'''
        super().__init__()
        self.conv1 = nn.Conv1d(params['conv1_in'], params['conv1_out'], params['conv1_ksize'], params['conv1_stride'])
        self.conv2 = nn.Conv1d(params['conv1_out'], params['conv2_out'], params['conv2_ksize'], params['conv2_stride'])
        self.conv3 = nn.Conv1d(params['conv2_out'], params['conv3_out'], params['conv3_ksize'], params['conv3_stride'])
        self.conv4 = nn.Conv1d(params['conv3_out'], params['conv4_out'], params['conv4_ksize'], params['conv4_stride'])
        self.pool1 = nn.MaxPool1d(kernel_size=params['pool1_ksize'])
        self.pool2 = nn.MaxPool1d(kernel_size=params['pool2_ksize'])
        self.pool3 = nn.MaxPool1d(kernel_size=params['pool3_ksize'])
        self.linear1 = nn.Linear(params['conv4_out'], params['conv4_out'])
        self.linear2 = nn.Linear(params['conv4_out'], 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=params['p_dropout'])
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class cnn2(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.conv1 = nn.Conv1d(params['conv1_in'], params['conv1_out'], params['conv1_ksize'], params['conv1_stride'])
        self.conv2 = nn.Conv1d(params['conv1_out'], params['conv2_out'], params['conv2_ksize'], params['conv2_stride'])
        self.conv3 = nn.Conv1d(params['conv2_out'], params['conv3_out'], params['conv3_ksize'], params['conv3_stride'])
        self.conv4 = nn.Conv1d(params['conv3_out'], params['conv4_out'], params['conv4_ksize'], params['conv4_stride'])
        self.pool1 = nn.MaxPool1d(kernel_size=params['pool1_ksize'])
        self.pool2 = nn.MaxPool1d(kernel_size=params['pool2_ksize'])
        self.pool3 = nn.MaxPool1d(kernel_size=params['pool3_ksize'])
        self.linear = nn.Linear(params['conv4_out'], 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=params['p_dropout'])

        # self.bn1 = nn.BatchNorm1d(20)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool2(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool3(x)

        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x

class cnn3(nn.Module):
    def __init__(self, params):
        '''
        {
        'conv1_in': 16,
        'conv1_out': 128,
        'conv1_ksize': 50,
        'conv1_stride': 5,
        'conv2_out': 256,
        'conv2_ksize': 10,
        'conv2_stride': 3,
        'conv3_out': 512,
        'conv3_ksize': 5,
        'conv3_stride': 1,
        'pool1_ksize': 4,
        'pool2_ksize': 2,
        'pool3_ksize': 2,
        'p_dropout': 0.6,
        }'''
        super().__init__()
        self.conv1 = nn.Conv1d(params['conv1_in'], params['conv1_out'], params['conv1_ksize'], params['conv1_stride'])
        self.conv2 = nn.Conv1d(params['conv1_out'], params['conv2_out'], params['conv2_ksize'], params['conv2_stride'])
        self.conv3 = nn.Conv1d(params['conv2_out'], params['conv3_out'], params['conv3_ksize'], params['conv3_stride'])
        self.pool1 = nn.MaxPool1d(kernel_size=params['pool1_ksize'])
        self.pool2 = nn.MaxPool1d(kernel_size=params['pool2_ksize'])
        self.pool3 = nn.MaxPool1d(kernel_size=params['pool3_ksize'])
        self.linear1 = nn.Linear(params['conv3_out'], params['conv3_out'])
        self.linear2 = nn.Linear(params['conv3_out'], params['out_features'])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=params['p_dropout'])

        self.bn1 = nn.BatchNorm1d(params['conv1_out'])
        self.bn2 = nn.BatchNorm1d(params['conv2_out'])
        self.bn3 = nn.BatchNorm1d(params['conv3_out'])

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class cnn_or(nn.Module):
    def __init__(self, params):
        '''
        {
        'conv1_in': 16,
        'conv1_out': 128,
        'conv1_ksize': 50,
        'conv1_stride': 5,
        'conv2_out': 256,
        'conv2_ksize': 10,
        'conv2_stride': 3,
        'conv3_out': 512,
        'conv3_ksize': 5,
        'conv3_stride': 1,
        'pool1_ksize': 4,
        'pool2_ksize': 2,
        'pool3_ksize': 2,
        'p_dropout': 0.6,
        }'''
        super().__init__()
        self.conv1 = nn.Conv1d(params['conv1_in'], params['conv1_out'], params['conv1_ksize'], params['conv1_stride'])
        self.conv2 = nn.Conv1d(params['conv1_out'], params['conv2_out'], params['conv2_ksize'], params['conv2_stride'])
        self.conv3 = nn.Conv1d(params['conv2_out'], params['conv3_out'], params['conv3_ksize'], params['conv3_stride'])
        self.pool1 = nn.MaxPool1d(kernel_size=params['pool1_ksize'])
        self.pool2 = nn.MaxPool1d(kernel_size=params['pool2_ksize'])
        self.pool3 = nn.MaxPool1d(kernel_size=params['pool3_ksize'])
        self.linear1 = nn.Linear(params['conv3_out'], params['conv3_out'])
        self.linear2 = nn.Linear(params['conv3_out'], params['out_features'])

        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=params['p_dropout'])

        self.bn1 = nn.BatchNorm1d(params['conv1_out'])
        self.bn2 = nn.BatchNorm1d(params['conv2_out'])
        self.bn3 = nn.BatchNorm1d(params['conv3_out'])

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

class cnn_2d_1(nn.Module):
    def __init__(self, params):
        '''
        {
            'in_channels': 1,
            'conv1_out': 32,
            'conv1_ksize': 16,
            'conv1_stride': 8,
            'conv2_out': 128,
            'conv2_ksize': 10,
            'conv2_stride': 7,
            'conv3_out': 512,
            'conv3_ksize': 3,
            'conv3_stride': 1,
            'pool1_ksize': 2,
            'pool2_ksize': 2,
            'pool3_ksize': 2,
            'p_dropout': 0.6,
            'out_features': 1
        }'''
        super().__init__()
        self.conv1 = nn.Conv2d(params['in_channels'], params['conv1_out'], params['conv1_ksize'], params['conv1_stride'])
        self.conv2 = nn.Conv1d(params['conv1_out'], params['conv2_out'], params['conv2_ksize'], params['conv2_stride'])
        self.conv3 = nn.Conv1d(params['conv2_out'], params['conv3_out'], params['conv3_ksize'], params['conv3_stride'])

        self.pool1 = nn.MaxPool1d(params['pool1_ksize'])
        # self.pool1 = nn.AdaptiveAvgPool1d(params['pool1_ksize'])
        self.pool2 = nn.MaxPool1d(params['pool2_ksize'])
        self.pool3 = nn.MaxPool1d(params['pool3_ksize'])

        self.bn1 = nn.BatchNorm1d(params['conv1_out'])
        self.bn2 = nn.BatchNorm1d(params['conv2_out'])
        self.bn3 = nn.BatchNorm1d(params['conv3_out'])
        self.bn4 = nn.BatchNorm1d(params['conv3_out'])

        self.linear1 = nn.Linear(params['conv3_out'], params['conv3_out'])
        self.linear2 = nn.Linear(params['conv3_out'], params['out_features'])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(params['p_dropout'])
    
    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape((x.shape[0], x.shape[1], x.shape[3])) # 2d->1dの変換
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.reshape((x.shape[0], -1))
        x = self.linear1(x)
        # x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        return x

class cnn_2d_2(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.conv1 = nn.Conv2d(params['in_channels'], params['conv1_out'], params['conv1_ksize'], params['conv1_stride'])
        self.conv2 = nn.Conv2d(params['conv1_out'], params['conv2_out'], params['conv2_ksize'], params['conv2_stride'])
        self.conv3 = nn.Conv2d(params['conv2_out'], params['conv3_out'], params['conv3_ksize'], params['conv3_stride'])
        self.conv4 = nn.Conv1d(params['conv3_out'], params['conv4_out'], params['conv4_ksize'], params['conv4_stride'])
        self.conv5 = nn.Conv1d(params['conv4_out'], params['conv5_out'], params['conv5_ksize'], params['conv5_stride'])
        self.conv6 = nn.Conv1d(params['conv5_out'], params['conv6_out'], params['conv6_ksize'], params['conv6_stride'])

        self.pool_k2 = nn.MaxPool1d(2)

        self.bn1 = nn.BatchNorm2d(params['conv1_out'])
        self.bn2 = nn.BatchNorm2d(params['conv2_out'])
        self.bn3 = nn.BatchNorm2d(params['conv3_out'])

        self.linear1 = nn.Linear(params['conv6_out'], params['conv6_out'])
        self.linear2 = nn.Linear(params['conv6_out'], 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(params['p_dropout'])
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.pool1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        # x = self.pool2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        # x = self.pool3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.reshape((x.shape[0], x.shape[1], x.shape[3])) # 2d->1dの変換

        x = self.conv4(x)
        x = self.pool_k2(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.pool_k2(x)
        x = self.relu(x)
        
        x = self.conv6(x)
        x = self.pool_k2(x)
        x = self.relu(x)
        
        x = x.reshape((x.shape[0], -1))
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        return x

class modeler_cnnf(model_torch_base.modeler_torch):
    def __init__(self, params, rand):
        super().__init__(params, rand)
        self.model_class = params['model']
        self.dataset_class = dataset_cnnf

        self.loss_fn = nn.L1Loss()
    
    def train(self, tr_x, tr_y, es_x, es_y):
        tr_dataset = self.dataset_class(tr_x, tr_y)
        es_dataset = self.dataset_class(es_x, es_y)
        super().train(tr_dataset, es_dataset)
    
    def predict(self, x):
        self.model.eval()

        x = torch.tensor(x, dtype=torch.float32)
        pred = self.model(x).detach().numpy()
        pred = max(x, 0)
        pred = min(x, 29)
        return pred
