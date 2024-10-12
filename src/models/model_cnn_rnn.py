import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__)))
import model_base, model_torch_base
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def transform_x(x):
    '''
    1000点から100点*30データを等間隔で切り出す'''
    x = np.array([[[c[31*i:100+31*i] for i in range(30)] for c in t] for t in x])   # 切り出し
    x = x.transpose(0,2,1,3)
    return x

class cnn_rnn_Dataset(Dataset):
    def __init__(self, x, y):
        '''
        xは(trials,30,16,100)'''
        x = transform_x(x)
        
        y = np.array(y)
        # y = y.transpose(0,2,1)

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]

class cnn_rnn(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.conv_layer1 = [nn.Conv1d(params['conv1_in'], params['conv1_out'], params['conv1_ksize'], params['conv1_stride']) for _ in range(16)]
        self.conv_layer2 = [nn.Conv1d(params['conv1_out'], params['conv2_out'], params['conv2_ksize'], params['conv2_stride']) for _ in range(16)]

        self.pool1 = nn.MaxPool1d(params['pool1_ksize'], params['pool1_stride'])
        self.pool2 = nn.MaxPool1d(params['pool2_ksize'], params['pool2_stride'])

        self.rnn = nn.LSTM(params['conv2_out']*16, params['rnn_hidden'], num_layers=params['rnn_n_layers'], dropout=params['rnn_dropout'])
        self.linear1 = nn.Linear(params['rnn_hidden'], params['linear_out'])

        self.dropout = nn.Dropout(params['p_dropout'])

    def forward(self, x):
        shape = x.shape
        x = x.reshape((-1,shape[2],shape[3]))

        # conv1
        x = [conv(x[:,i,:].reshape((x.shape[0],1,x.shape[2]))) for i, conv in enumerate(self.conv_layer1)]
        x = [self.pool1(i) for i in x]
        x = [self.dropout(i) for i in x]

        # conv2
        x = [conv(x[i]) for i, conv in enumerate(self.conv_layer2)]
        x = [self.pool2(i) for i in x]
        x = [self.dropout(i) for i in x]

        # concat
        x = torch.cat(x,1)
        x = x.reshape((shape[0],shape[1],-1))

        # rnn
        x, _ = self.rnn(x)
        x = self.dropout(x)

        # affine
        x = self.linear1(x)        
        return x

class cnn_rnn2(nn.Module):
    def __init__(self, params):
        '''
        {
        'conv1_in': 16,
        'conv1_out': 128,
        'conv1_ksize': 50,
        'conv1_stride': 8,
        'conv1_padding': 1,
        'conv2_out': 256,
        'conv2_ksize': 3,
        'conv2_stride': 1,
        'conv2_padding': 1,
        'pool1_ksize': 2,
        'pool2_ksize': 2,
        'rnn_hidden': 1024,
        'out_features': 1,
        'p_dropout': 0.6
        }'''
        super().__init__()
        self.conv1 = nn.Conv1d(params['conv1_in'], params['conv1_out'], params['conv1_ksize'], params['conv1_stride'], params['conv1_padding'])
        self.conv2 = nn.Conv1d(params['conv1_out'], params['conv2_out'], params['conv2_ksize'], params['conv2_stride'], params['conv2_padding'])
        
        self.pool1 = nn.MaxPool1d(params['pool1_ksize'])
        self.pool2 = nn.MaxPool1d(params['pool2_ksize'])

        self.rnn = nn.LSTM(params['conv2_out'], params['rnn_hidden'])

        self.linear1 = nn.Linear(params['rnn_hidden'], params['rnn_hidden'])
        self.linear2 = nn.Linear(params['rnn_hidden'], params['out_features'])

        self.bn1 = nn.BatchNorm1d(params['conv1_out'])
        self.bn2 = nn.BatchNorm1d(params['conv2_out'])
        self.bn3 = nn.BatchNorm1d(params['rnn_hidden'])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(params['p_dropout'])
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(x)

        # x = torch.permute(x, (0,2,1))
        # x = torch.permute(x, (0,2,1))

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = torch.permute(x, (0,2,1))

        x, _ = self.rnn(x)
        
        x = torch.permute(x, (0,2,1))
        x = self.bn3(x)
        x = torch.permute(x, (0,2,1))
        x = self.dropout(x)

        # x = self.linear1(x)
        # x = self.relu(x)

        x = self.linear2(x)
        x = x.reshape((x.shape[0], x.shape[1]))
        return x

class cnn_rnn3(nn.Module):
    def __init__(self, params):
        '''
        {
        'conv1_in': 16,
        'conv1_out': 128,
        'conv1_ksize': 50,
        'conv1_stride': 5,
        'conv1_padding': 5,
        'conv2_out': 256,
        'conv2_ksize': 10,
        'conv2_stride': 3,
        'conv2_padding': 1,
        'conv3_out': 512,
        'conv3_ksize': 3,
        'conv3_stride': 1,
        'conv3_padding': 1,
        'pool1_ksize': 2,
        'rnn_hidden': 1024,
        'out_features': 1,
        'p_dropout': 0.5,
        }'''
        super().__init__()

        self.conv1 = nn.Conv1d(params['conv1_in'], params['conv1_out'], params['conv1_ksize'], params['conv1_stride'], params['conv1_padding'])
        self.conv2 = nn.Conv1d(params['conv1_out'], params['conv2_out'], params['conv2_ksize'], params['conv2_stride'], params['conv2_padding'])
        self.conv3 = nn.Conv1d(params['conv2_out'], params['conv3_out'], params['conv3_ksize'], params['conv3_stride'], params['conv3_padding'])
        
        self.pool1 = nn.MaxPool1d(params['pool1_ksize'])
        self.pool2 = nn.MaxPool1d(params['pool2_ksize'])

        self.rnn = nn.LSTM(params['conv3_out'], params['rnn_hidden'], num_layers=params['rnn_layers'], dropout=params['rnn_dropout'])

        self.linear1 = nn.Linear(params['rnn_hidden'], params['rnn_hidden'])
        self.linear2 = nn.Linear(params['rnn_hidden'], params['out_features'])

        self.bn1 = nn.BatchNorm1d(params['conv1_out'])
        self.bn2 = nn.BatchNorm1d(params['conv2_out'])
        self.bn3 = nn.BatchNorm1d(params['conv3_out'])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(params['p_dropout'])
    
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
        x = self.bn3(x)
        x = self.dropout(x)
        x = self.relu(x)
        
        x = torch.permute(x, (0,2,1))
        x, _ = self.rnn(x)
        x = self.dropout(x)

        x = self.linear2(x)

        x = x.reshape((x.shape[0], x.shape[1]))
        return x

class modeler_cnn_rnn(model_torch_base.modeler_torch):
    def __init__(self, params, rand):
        super().__init__(params, rand)
        self.model_class = params['model_class']
        self.dataset_class = model_torch_base.dataset_ndarray

        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.L1Loss()
        # self.loss_fn = model_torch_base.RMSELoss()
    
    def train(self, tr_x, tr_y, es_x, es_y):
        tr_dataset = self.dataset_class(tr_x, tr_y)
        es_dataset = self.dataset_class(es_x, es_y)

        super().train(tr_dataset, es_dataset)

    def predict(self, x):
        self.model.eval()

        # x = transform_x(x)
        x = torch.tensor(x, dtype=torch.float32)

        pred = self.model(x).detach().numpy().flatten().astype(np.float64)
        return pred

if __name__=='__main__':
    train = sio.loadmat(config.train_raw_path)
    test = sio.loadmat(config.test_raw_path)

    train_myo = model_torch_base.myo_processor(train, 50)
    test_myo = model_torch_base.myo_processor(test, 50)
    y = model_torch_base.vel_extractor(train)

    model_params = {
        'conv1_in': 16,
        'conv1_out': 128,
        'conv1_ksize': 50,
        'conv1_stride': 5,
        'conv1_padding': 5,
        'conv2_out': 256,
        'conv2_ksize': 10,
        'conv2_stride': 3,
        'conv2_padding': 1,
        'conv3_out': 512,
        'conv3_ksize': 3,
        'conv3_stride': 1,
        'conv3_padding': 1,
        'pool1_ksize': 2,
        'pool2_ksize': 1,
        'rnn_hidden': 128,
        'rnn_layers': 1,
        'rnn_dropout': 0,
        'out_features': 1,
        'p_dropout': 0.5,
    }

    params = {
        'modeltype': 'cnn-lstm',
        'rand': 1,
        'use_cv': False,
        'normalize': True,
        'smoothing': True,
        'verbose': True,
        'split_by_subject': False,
        'modeler_params': {
            'model_class': cnn_rnn3,
            'num_epoch': 50,
            'estop_epoch': 50,
            'batch_size': 10,
            'lr': 1e-3,
            'verbose': False,
            'model_params': model_params
        }
    }

    vp = model_torch_base.vel_prediction_ndarray(modeler_cnn_rnn, params)
    vp.main(train_myo, y, test_myo)