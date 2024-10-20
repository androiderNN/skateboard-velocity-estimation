import os, sys, math
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__)))
import model_base, model_torch_base
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class cnn1(nn.Module):
    def __init__(self, params):
        '''
        {
            'conv1_in': 16,
            'conv1_out': 16,
            'conv1_ksize': 50,
            'conv1_stride': 10,
            'conv2_out': 64,
            'conv2_ksize': 10,
            'conv2_stride': 3,
            'conv3_out': 128,
            'conv3_ksize': 3,
            'conv3_stride': 1,
            'pool1_ksize': 2,
            'pool1_padding': 0,
            'pool2_ksize': 2,
            'pool2_padding': 0,
            'p_dropout': 0.5
        }'''
        super().__init__()
        
        params = {
            'conv1_in': 16,
            'conv1_out': 16,
            'conv1_ksize': 50,
            'conv1_stride': 10,
            'conv2_out': 16,
            'conv2_ksize': 10,
            'conv2_stride': 3,
            'conv3_out': 16,
            'conv3_ksize': 3,
            'conv3_stride': 1,
            'pool1_ksize': 2,
            'pool1_padding': 0,
            'pool2_ksize': 2,
            'pool2_padding': 0,
            'p_dropout': 0.5
        }

        self.conv1 = nn.Conv1d(params['conv1_in'], params['conv1_out'], params['conv1_ksize'], params['conv1_stride'])
        self.conv2 = nn.Conv1d(params['conv1_out'], params['conv2_out'], params['conv2_ksize'], params['conv2_stride'])
        self.conv3 = nn.Conv1d(params['conv2_out'], params['conv3_out'], params['conv3_ksize'], params['conv3_stride'])

        self.pool1 = nn.MaxPool1d(params['pool1_ksize'], padding=params['pool1_padding'])
        self.pool2 = nn.MaxPool1d(params['pool2_ksize'], padding=params['pool2_padding'])

        self.linear = nn.Linear(params['conv3_out']*4, 30)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(params['p_dropout'])

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = x.reshape((x.shape[0], -1))
        x = self.linear(x)

        return x

class cnn2(nn.Module):
    def __init__(self, params):
        '''     
        {
            'conv1_in': 16,
            'conv1_out': 64,
            'conv1_ksize': 50,
            'conv1_stride': 10,
            'conv2_out': 128,
            'conv2_ksize': 10,
            'conv2_stride': 3,
            'conv3_out': 256,
            'conv3_ksize': 3,
            'conv3_stride': 1,
            'conv4_out': 1024,
            'conv4_ksize': 2,
            'conv4_stride': 1,
            'pool1_ksize': 2,
            'pool1_padding': 0,
            'pool2_ksize': 2,
            'pool2_padding': 0,
            'pool3_ksize': 2,
            'pool3_padding': 0,
            'p_dropout': 0.5
        }'''
        super().__init__()
        
        self.conv1 = nn.Conv1d(params['conv1_in'], params['conv1_out'], params['conv1_ksize'], params['conv1_stride'])
        self.conv2 = nn.Conv1d(params['conv1_out'], params['conv2_out'], params['conv2_ksize'], params['conv2_stride'])
        self.conv3 = nn.Conv1d(params['conv2_out'], params['conv3_out'], params['conv3_ksize'], params['conv3_stride'])
        self.conv4 = nn.Conv1d(params['conv3_out'], params['conv4_out'], params['conv4_ksize'], params['conv4_stride'])

        self.pool1 = nn.MaxPool1d(params['pool1_ksize'])
        self.pool2 = nn.MaxPool1d(params['pool2_ksize'])
        self.pool3 = nn.MaxPool1d(params['pool3_ksize'])

        self.linear = nn.Linear(params['conv4_out'], 30)

        self.bn1 = nn.BatchNorm1d(params['conv1_out'])
        self.bn2 = nn.BatchNorm1d(params['conv2_out'])
        self.bn3 = nn.BatchNorm1d(params['conv3_out'])

        self.dropout = nn.Dropout(params['p_dropout'])
        self.relu = nn.ReLU()

        self.n = params['conv4_out']

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = x.reshape((x.shape[0], self.n))
        x = self.linear(x)
        return x

class modeler_cnn(model_torch_base.modeler_torch):
    def __init__(self, params, rand):
        super().__init__(params, rand)
        self.model_class = params['model_class']
        self.dataset_class = model_torch_base.dataset_ndarray

        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.L1Loss()
        # self.loss_fn = RMSELoss()

    def train(self, tr_x, tr_y, es_x, es_y):
        self.params['model_params']['input_size'] = tr_x.shape[1]

        tr_dataset = self.dataset_class(tr_x, tr_y)
        es_dataset = self.dataset_class(es_x, es_y)

        super().train(tr_dataset, es_dataset)

    def predict(self, x):
        self.model.eval()

        x = torch.tensor(x, dtype=torch.float32)

        pred = self.model(x).detach().numpy().flatten().astype(np.float64)
        return pred

if __name__=='__main__':
    train = sio.loadmat(config.train_raw_path)
    test = sio.loadmat(config.test_raw_path)

    train_myo = model_torch_base.myo_processor(train, 5)
    test_myo = model_torch_base.myo_processor(test, 5)
    y = model_torch_base.vel_extractor(train)

    model_params = {
        'conv1_in': 16,
        'conv1_out': 64,
        'conv1_ksize': 50,
        'conv1_stride': 10,
        'conv2_out': 128,
        'conv2_ksize': 10,
        'conv2_stride': 3,
        'conv3_out': 256,
        'conv3_ksize': 3,
        'conv3_stride': 1,
        'conv4_out': 64,
        'conv4_ksize': 2,
        'conv4_stride': 1,
        'pool1_ksize': 2,
        'pool1_padding': 0,
        'pool2_ksize': 2,
        'pool2_padding': 0,
        'pool3_ksize': 2,
        'pool3_padding': 0,
        'p_dropout': 0.3
    }

    params = {
        'modeltype': 'cnn',
        'rand': 0,
        'use_cv': False,
        'normalize': False,
        'smoothing': False,
        'verbose': True,
        'split_by_subject': False,
        'modeler_params': {
            'model_class': cnn2,
            'num_epoch': 1000,
            'estop_epoch': 100,
            'batch_size': 10,
            'lr': 1e-3,
            'verbose': False,
            'model_params': model_params
        }
    }

    vp = model_torch_base.vel_prediction_ndarray(modeler_cnn, params)
    vp.main(train_myo, y, test_myo)