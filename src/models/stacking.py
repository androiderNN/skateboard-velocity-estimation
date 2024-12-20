import os, pickle, datetime, json, sys, math, datetime
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

def getmean(pred):
    pred = np.array(pred)

    pred = pred.reshape((pred.shape[0], int(pred.shape[1]/30), 30, pred.shape[2]))  # (num_preds, 1200, 30, 9ぐらい)
    pmean = pred.mean(axis=0)   # (1200, 30, 9ぐらい)

    # z軸方向の速度が0のデータについてxの変動が大きいデータの削除
    zmax = pmean[:,:,-1].max(axis=1)
    zmin = pmean[:,:,-1].min(axis=1)
    zdiff = zmax - zmin
    zmask = (zdiff<0.5) & (zmax<0.5) & (zmin>-0.5)   # z方向の速度がほぼ0のtrialのmask

    # 各trialでxの速度が条件を満たす予測値のみを用いて平均値を取り直す
    for i in range(pmean.shape[0]):
        if zmask[i]:    # zの速度が0周辺
            x = pred[:,i,:,-3]
            
            xdiff = x.max(axis=1) - x.min(axis=1)
            xmask1 = xdiff < 1.5   # xのdiffが1.5未満

            # zが0付近の時は平面上の滑走なので速度xの絶対値は一般に大きい
            xmask2 = abs(x).min(axis=1) > 2

            xmask = xmask1 & xmask2 # maskの統合
            
            if sum(xmask)>0:    # 条件を満たすxの予測値が存在する場合
                tmp = pred[xmask,i,:,-3]
                pmean[i,:,-3] = tmp.mean(axis=0)
    
    pmean = pmean.reshape(-1,pmean.shape[2])
    
    return pmean

def meanstack():
    stack_ids = [
        # 'lgb_1027_01:55:57',
        # 'lgb_1027_01:57:32',
        # 'lgb_1027_01:59:33',
        # 'lgb_1027_02:01:09',
        'lstm_1027_07:24:15',
        'lstm_1027_07:33:41',
        'lstm_1027_07:43:26',
        'lstm_1027_08:00:30',
    ]
    train_pred = [pickle.load(open(os.path.join(config.exdir, id, 'train_pred.pkl'), 'rb')) for id in stack_ids]
    test_pred = [pickle.load(open(os.path.join(config.exdir, id, 'test_pred.pkl'), 'rb')) for id in stack_ids]

    tr_col = train_pred[0].columns
    te_col = test_pred[0].columns

    # 平均値の算出
    train_pred = getmean(train_pred)
    test_pred = getmean(test_pred)

    train_pred = pd.DataFrame(train_pred, columns=tr_col)
    test_pred = pd.DataFrame(test_pred, columns=te_col)

    tr_rmse = model_base.rmse_3d(train_pred)
    print('\ntrain rmse :', tr_rmse)

    now = datetime.datetime.now()
    time = now.strftime('%m%d_%H:%M:%S')

    dirpath = model_base.makeexportdir('meanstack', time, False)
    os.mkdir(dirpath)
    test_pred = test_pred.astype({'sid': int, 'trial': int, 'timepoint': int})
    model_base.make_submission(test_pred, dirpath)
    pickle.dump(train_pred, open(os.path.join(dirpath, 'train_pred.pkl'), 'wb'))
    pickle.dump(test_pred, open(os.path.join(dirpath, 'test_pred.pkl'), 'wb'))

    print('finished')

class model_stack_cnn(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.conv = nn.Conv1d(params['in_channels'], 30, 30)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(params['p_dropout'])
        self.linear = nn.Linear(30, 30)
    
    def forward(self, x):
        x = self.conv(x)

        # x = self.dropout(x)
        # x = self.relu(x)

        # x = x.reshape(x.shape[:2])
        # x = self.linear(x)
        return x

class stack_cnn():
    def __init__(self, params):
        self.stack_models = {
            # 'lgb': 'lgb_1015_07:11:19_cv',
            # 'cnn': 'cnn_1015_07:13:48_cv',
            # 'lstm': 'lstm_1015_07:55:58_cv',
            # 'cnn_lstm': 'cnn-lstm_1013_19:26:48',

            # 'lgb1': 'lgb_1027_01:55:57',
            # 'lgb2': 'lgb_1027_01:57:32',
            # 'lgb3': 'lgb_1027_01:59:33',
            # 'lgb4': 'lgb_1027_02:01:09',
            'lstm1': 'lstm_1027_07:24:15',
            'lstm2': 'lstm_1027_07:33:41',
            'lstm3': 'lstm_1027_07:43:26',
            'lstm4': 'lstm_1027_08:00:30',
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
        # train_preds = {k: pickle.load(open(os.path.join(config.exdir, self.stack_models[k], 'train_cv_pred.pkl'), 'rb')) for k in self.stack_models.keys()}
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

            if self.params['use_cv']:
                trainer = model_torch_base.cv_training_ndarray(self.modeler_class, self.params, self.score_fn)
            else:
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
        self.expath = model_base.makeexportdir(self.params['modeltype'], time, self.params['use_cv'])

        if self.params['verbose']:
            self.exornot = input('\n出力しますか(y/n)')=='y'

        if self.exornot:
            os.mkdir(self.expath)   # 出力日時記載のフォルダ作成
            model_base.make_submission(self.test_pred.copy(), self.expath)
            pickle.dump(self.train_pred, open(os.path.join(self.expath, 'train_pred.pkl'), 'wb'))
            pickle.dump(self.test_pred, open(os.path.join(self.expath, 'test_pred.pkl'), 'wb'))
            pickle.dump(self.params, open(os.path.join(self.expath, 'params.pkl'), 'wb'))
