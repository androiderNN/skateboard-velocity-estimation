import os, pickle, datetime, json, sys, math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
import model_base, model_torch_base, model_cnn

now = datetime.datetime.now()
time = now.strftime('%m%d_%H:%M:%S')

class model_stack_cnn(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(params['in_channels'], 30, 30)
        )
    
    def forward(self, x):
        x = self.net(x)
        x = x.reshape(x.shape[:2])
        # print(x.shape)
        return x

class stack():
    def __init__(self, params):
        self.stack_models = {
            'lgb': 'lgb_1013_16:02:36',
            'cnn': 'cnn_1013_15:49:46',
            'lstm': 'lstm_1013_16:03:15',
            # 'cnn_lstm': 'cnn-lstm_1012_17:41:42',
        }

        self.params = params
        self.params['modeler_params']['model_params']['in_channels'] = len(self.stack_models.keys())

        self.modeler_class = model_cnn.modeler_cnn
        self.score_fn = model_base.rmse

        self.train_pred = pickle.load(open(config.train_path, 'rb')).loc[:, ['sid', 'trial', 'timepoint', 'vel_x', 'vel_y', 'vel_z']]
        self.test_pred = pickle.load(open(config.test_path, 'rb')).loc[:, ['sid', 'trial', 'timepoint']]

    def dataset(self):
        '''
        {target: [(tr_trials,30,n_models)*2, (te_trials,30,n_models)]}'''
        train_preds = {k: pickle.load(open(os.path.join(config.exdir, self.stack_models[k], 'train_pred.pkl'), 'rb')) for k in self.stack_models.keys()}
        test_preds = {k: pickle.load(open(os.path.join(config.exdir, self.stack_models[k], 'test_pred.pkl'), 'rb')) for k in self.stack_models.keys()}

        self.data = dict()

        for target in config.target_name:
            tr_x = np.array([train_preds[k][target+'_pred'] for k in train_preds.keys()])
            tr_x = tr_x.reshape((len(self.stack_models.keys()), -1, 30)).transpose(1,0,2)

            te_x = np.array([test_preds[k][target+'_pred'] for k in test_preds.keys()])
            te_x = te_x.reshape((len(self.stack_models.keys()), -1, 30)).transpose(1,0,2)

            tr_y = np.array(train_preds[list(train_preds.keys())[0]][target])
            tr_y = tr_y.reshape((-1, 30))

            self.data[target] = [tr_x, tr_y, te_x]

    def main(self):
        self.dataset()

        for target in config.target_name:
            print('\n', target)

            tr_x, tr_y, te_x = self.data[target]

            trainer = model_torch_base.holdout_training_ndarray(self.modeler_class, self.params, self.score_fn)
            tr_pred, te_pred = trainer.main(tr_x, tr_y, te_x)

            self.train_pred[target+'_pred'] = tr_pred
            self.test_pred[target+'_pred'] = te_pred

        # 平滑化
        if self.params['smoothing'] == 'ma':
            self.train_pred = model_base.smoothing_movingaverage(self.train_pred, ksize=5)
            self.test_pred = model_base.smoothing_movingaverage(self.test_pred, ksize=5)
        elif self.params['smoothing'] == 'lp':
            self.train_pred = model_base.smoothing_lowpass(self.train_pred, lowcut=1)
            self.test_pred = model_base.smoothing_lowpass(self.test_pred, lowcut=1)
        elif self.params['smoothing'] == False:
            pass
        else:
            raise ValueError
        
        # 出力
        index = train_test_split([i for i in range(int(self.train_pred.shape[0]/30))], test_size=0.2, random_state=self.params['rand'])
        index = [[i*30+j for i in idx for j in range(30)] for idx in index]
        tr_rmse = model_base.rmse_3d(self.train_pred.iloc[index[0],:])
        print('\ntrain rmse :', tr_rmse)
        es_rmse = model_base.rmse_3d(self.train_pred.iloc[index[1],:])
        print('validation rmse :', es_rmse)

        # 保存
        self.expath = model_base.makeexportdir(self.params['modeltype'], time)

        if self.params['verbose']:
            self.exornot = input('\n出力しますか(y/n)')=='y'

        if self.exornot:
            os.mkdir(self.expath)   # 出力日時記載のフォルダ作成
            model_base.make_submission(self.test_pred.copy(), self.expath)
            pickle.dump(self.train_pred, open(os.path.join(self.expath, 'train_pred.pkl'), 'wb'))
            pickle.dump(self.test_pred, open(os.path.join(self.expath, 'test_pred.pkl'), 'wb'))
            pickle.dump(self.params, open(os.path.join(self.expath, 'params.pkl'), 'wb'))
