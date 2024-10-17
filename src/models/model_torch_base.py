import os, sys, pickle, copy, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold

sys.path.append(os.path.join(os.path.dirname(__file__)))
import model_base
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'features'))
import iemg

now = datetime.datetime.now()
time = now.strftime('%m%d_%H:%M:%S')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def myo_processor(raw_data, lowcut=999):
    myo = np.array([t for s in range(1,5) for t in raw_data[str(s).zfill(4)][0,0][0]])   # 筋電位データのみ抽出 (12**, 16, 1000)
    myo = abs(myo)

    if lowcut < 999:    # lowcutが999未満のときフィルタリング
        myo = np.array([iemg.apply_filter(t, lowcut) for t in myo])
    else:   # 999のときはここで正規化
        myo = myo / myo.max(axis=2)[:,:,np.newaxis]
    
    return myo

def vel_extractor(train_raw):
    vel = [t for s in range(1,5) for t in train_raw[str(s).zfill(4)][0,0][1]]
    vel = np.array(vel)
    return vel

class dataset_df(Dataset):
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

class dataset_ndarray(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, y):
        return ((pred - y)**2).mean() **0.5

class modeler_torch(model_base.modeler_base):
    '''
    predictメソッドの定義'''
    def __init__(self, params, rand):
        '''
        model_class, dataset_class, loss_fnが必要'''
        self.params = params
        self.rand = rand

        self.model_class = None
        self.dataset_class = None
        self.model = None
        self.optimizer = None
        self.loss_fn = None

        self.log = None
        self.best_model = None
    
    def train_loop(self, dataloader):
        self.model.train()
        for batch, (x, y) in enumerate(dataloader):
            pred = self.model(x)
            loss = self.loss_fn(pred.flatten(), y.flatten())

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
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.99)

        # データローダー
        train_dataloader = DataLoader(train_dataset, batch_size=self.params['batch_size'], shuffle=True)
        estop_dataloader = DataLoader(estop_dataset, batch_size=self.params['batch_size'], shuffle=True)

        # 記録用ndarray
        self.log = list()

        for epoch in range(self.params['num_epoch']):
            self.train_loop(train_dataloader)

            self.log.append([0,0])
            self.log[epoch][0] = self.test_loop(train_dataloader)
            self.log[epoch][1] = self.test_loop(estop_dataloader)

            scheduler.step()

            if epoch%10==0:
                print(f'estop rmse: {self.log[-1][1]} [{epoch}/{self.params["num_epoch"]}]')

            # estopのスコアが向上していればbest_modelを現在のモデルに更新
            if self.log[-1][1] == min([l[1] for l in self.log]):
                self.best_model = copy.deepcopy(self.model.state_dict())
            
            # epochがestop_epoch+1回以上回り、かつロスがestop_epoch回前より落ちていなければ打ち切り
            if (len(self.log) > self.params['estop_epoch']) and (self.log[-1][1] > self.log[-self.params['estop_epoch']-1][1]):
                print(f'epoch {epoch} early stop')
                break
        
        self.model.load_state_dict(self.best_model)
        self.log = np.array(self.log)

        # 学習曲線描画
        if self.params['verbose']:
            plt.figure(figsize=(5,3))
            plt.xlabel('epochs')
            plt.ylabel('rmse')
            plt.plot(self.log)
            plt.show()

class holdout_training_ndarray():
    def __init__(self, modeler, params, score_fn):
        '''
        hold-outで学習・予測・スコア算出を行うクラス
        modeler : modeler_baseを継承したクラス
        score_fn : y, y_predを入力するとスカラーのスコアを返す関数'''
        self.modeler = modeler(params=params['modeler_params'], rand=params['rand'])
        self.params = params
        self.score_fn = score_fn

    def main(self, x, y, test):
        '''
        関数内でインデックス分割、学習、スコア出力、testデータの予測出力まで行う'''
        # データ分割
        tr_x, es_x, tr_y, es_y = train_test_split(x, y, test_size=0.2, random_state=self.params['rand'])

        # 学習
        self.modeler.train(tr_x, tr_y, es_x, es_y)

        # スコア出力
        tr_pred = self.modeler.predict(tr_x)
        tr_score = self.score_fn(tr_y.flatten(), tr_pred)
        print('train score :', tr_score)
        es_pred = self.modeler.predict(es_x)
        es_score = self.score_fn(es_y.flatten(), es_pred)
        print('estop score :', es_score)

        # 予測
        train_pred = self.modeler.predict(x)
        test_pred = self.modeler.predict(test)
        return train_pred, test_pred

    def predict(self, x):
        return self.modeler.predict(x)

class cv_training_ndarray():
    def __init__(self, modeler, params, score_fn):
        self.mdr = modeler
        self.modelers = list()
        self.params = params
        self.score_fn = score_fn

    def predict(self, x):
        '''
        xを投げると各foldのモデルで予測し平均値を予測値として返す'''
        pred = list()

        for modeler in self.modelers:
            pred.append(modeler.predict(x))
        
        pred = np.array(pred)
        pred = pred.mean(axis=0)
        return pred

    def main(self, x, y, test):
        '''
        trainデータ、特徴量の列名リスト、ターゲットの列名、テストデータを投げると学習と結果出力を行いtestデータの予測値を返す'''
        # valid分割
        tr_idx, va_idx = train_test_split([i for i in range(x.shape[0])], test_size=0.2, random_state=self.params['rand'])
        tr_x, tr_y = x[tr_idx], y[tr_idx]
        va_x, va_y = x[va_idx], y[va_idx]

        # fold分割
        kf = KFold(n_splits=4, shuffle=True, random_state=self.params['rand'])

        for i, (tr_index, es_index) in enumerate(kf.split(tr_x, tr_y)):
            print('\nFold', i+1)
            fold_x, fold_y = tr_x[tr_index], tr_y[tr_index]    # fold内での学習用データ
            es_x, es_y = tr_x[es_index], tr_y[es_index]    # fold内でのearly stopping用データ

            # 学習
            modeler = self.mdr(params=self.params['modeler_params'], rand=self.params['rand'])
            modeler.train(fold_x, fold_y, es_x, es_y)

            # スコア出力
            fold_pred = modeler.predict(fold_x)
            fold_score = self.score_fn(fold_y.flatten(), fold_pred.flatten())
            print('train score :', fold_score)

            es_pred = modeler.predict(es_x)
            es_score = self.score_fn(es_y.flatten(), es_pred.flatten())
            print('estop score :', es_score)

            self.modelers.append(modeler)

        # スコア出力
        print('\nmean prediction')
        train_pred = self.predict(tr_x)
        train_score = self.score_fn(tr_y.flatten(), train_pred)
        print('train score :', train_score)

        valid_pred = self.predict(va_x)
        valid_score = self.score_fn(va_y.flatten(), valid_pred)
        print('valid score :', valid_score, '\n')

        # testデータの予測
        train_pred = self.predict(x)
        test_pred = self.predict(test)
        
        # 記録用dataframe
        # 列は(fold, fold0~3の予測値, 全体の予測値, 学習に用いられていないモデルでの予測値)
        tmp = pickle.load(open(config.train_path, 'rb'))[['sid', 'trial', 'timepoint']+config.target_name]

        result = np.full((len(tmp), 7), 999, dtype=np.float64)
        result[:, 5] = train_pred

        va_index = [30*i+j for i in va_idx for j in range(30)]
        result[va_index, 6] = train_pred[va_index]  # validationは全体の予測値で埋める

        l = np.array([i for i in range(x.shape[0])])
        tr_index = l[tr_idx]    # validationを除いたindex

        for i, idx in enumerate(kf.split(tr_index)):
            es_idx = [30*i+j for i in tr_index[idx[1]] for j in range(30)]
            result[es_idx, 0] = i # foldの記録
            result[:, i+1] = self.modelers[i].predict(x)
            result[es_idx, 6] = result[es_idx, i+1]
        
        self.cv_pred = pd.DataFrame(result, columns=['fold'] + ['fold'+str(i)+'_pred' for i in range(4)] + ['mean_pred', 'valid_pred'])
        self.cv_pred = self.cv_pred.join(tmp)

        return train_pred, test_pred

class vel_prediction_ndarray():
    def __init__(self, modeler, params):
        self.params = params

        self.modeler = modeler
        self.expath = None

        self.trainer_array = list()
        self.train_pred = pickle.load(open(config.train_path, 'rb')).loc[:, ['sid', 'trial', 'timepoint', 'vel_x', 'vel_y', 'vel_z']]
        self.test_pred = pickle.load(open(config.test_path, 'rb')).loc[:, ['sid', 'trial', 'timepoint']]

    def main(self, x, y, test):
        '''
        y: (trial, 3, 30)'''
        # trainerの定義
        if self.params['use_cv']: # cross validation
            trainer_class = cv_training_ndarray
        else:   # hold out
            trainer_class = holdout_training_ndarray

        # x, y, zごとのモデル作成と予測
        for i, target in enumerate(config.target_name):
            print('\ntarget :', target)
            target_y = y[:,i,:]

            if not self.params['split_by_subject']:   # 被験者で分割しない場合
                # trainerの定義 cv_trainerまたはholdout_trainer
                trainer = trainer_class(self.modeler, self.params, score_fn=model_base.rmse)

                # 学習・予測
                tr_pred, te_pred = trainer.main(x, target_y, test)
                self.train_pred[target+'_pred'] = tr_pred
                self.test_pred[target+'_pred'] = te_pred

                self.trainer_array.append(trainer)
                # del trainer # jupyterが落ちるのでtrainerを消してみる

        # 予測値の平滑化
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

        # rmse出力 cvかhoかでvalidデータの使用目的（estop/valid)が異なるため注意
        index = train_test_split([i for i in range(x.shape[0])], test_size=0.2, random_state=self.params['rand'])
        index = [[i*30+j for i in idx for j in range(30)] for idx in index]
        tr_rmse = model_base.rmse_3d(self.train_pred.iloc[index[0],:])
        print('\ntrain rmse :', tr_rmse)
        es_rmse = model_base.rmse_3d(self.train_pred.iloc[index[1],:])
        print('validation rmse :', es_rmse)

        #保存
        self.expath = model_base.makeexportdir(self.params['modeltype'], time, self.params['use_cv'])

        if self.params['verbose']:
            if input('\n出力しますか(y/n)')=='y':
                os.mkdir(self.expath)   # 出力日時記載のフォルダ作成
                model_base.make_submission(self.test_pred.copy(), self.expath)
                pickle.dump(self.train_pred, open(os.path.join(self.expath, 'train_pred.pkl'), 'wb'))
                pickle.dump(self.test_pred, open(os.path.join(self.expath, 'test_pred.pkl'), 'wb'))
                pickle.dump(self.params, open(os.path.join(self.expath, 'params.pkl'), 'wb'))

                if self.params['use_cv']:   # foldごとの予測値保存
                    cv_pred = [tn.cv_pred for tn in self.trainer_array]
                    cv_pred_df = cv_pred[0][['sid', 'trial', 'timepoint']]

                    dic = {
                        'vel_x': cv_pred[0]['vel_x'],
                        'vel_x_pred': cv_pred[0]['valid_pred'],
                        'vel_y': cv_pred[1]['vel_y'],
                        'vel_y_pred': cv_pred[1]['valid_pred'],
                        'vel_z': cv_pred[2]['vel_z'],
                        'vel_z_pred': cv_pred[2]['valid_pred'],
                    }
                    cv_pred_df = cv_pred_df.join(pd.DataFrame(dic))

                    if self.params['smoothing'] == 'ma':
                        cv_pred_df = model_base.smoothing_movingaverage(cv_pred_df, ksize=5)
                    elif self.params['smoothing'] == 'lp':
                        cv_pred_df = model_base.smoothing_lowpass(cv_pred_df, lowcut=1)

                    pickle.dump(cv_pred_df, open(os.path.join(self.expath, 'train_cv_pred.pkl'), 'wb'))
            
                if input('モデルの保存(y/n)')=='y':
                    pickle.dump(self.trainer_array, open(os.path.join(config.saved_model_dir, self.params['modeltype']+'_'+time+'.pkl'), 'wb'))
                    print('model saved')
