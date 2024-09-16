import os, pickle, datetime, json, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

def get_tr_va_index(train, es_size=0.1, va_size=0, rand=0):
    '''
    trainのデータフレームを投げるとtrain, estop, validのindexを作成'''
    if es_size+va_size>1:
        raise ValueError
    
    sid = train['sid'].tolist()
    trial = train['trial'].tolist()
    train['trial_id'] = [str(sid[i])+'_'+str(trial[i]) for i in range(len(train))]    # sidとtrialでtrial_idを作成
    trial_id = np.unique(np.array(train['trial_id']))
    tr_id, es_id = train_test_split(trial_id, test_size=es_size, random_state=rand, shuffle=True)

    tr_index = [id in tr_id for id in train['trial_id']]
    es_index = [id in es_id for id in train['trial_id']]
    index = [tr_index, es_index]

    if va_size>0:   # validationが設定されているとき
        tr_id, va_id = train_test_split(tr_id, test_size=va_size, random_state=rand, shuffle=True)
        tr_index = [id in tr_id for id in train['trial_id']]
        va_index = [id in va_id for id in train['trial_id']]
        index = [tr_index, es_index, va_index]

    train.drop(columns='trial_id', inplace=True)    # trainは参照なのでtrial_idを削除しておく

    return index

def print_score(tr_y, tr_pred, es_y, es_pred, va_y, va_pred):
    '''
    正解と予測値を渡すとスコアを出力する'''
    print('train :', mean_squared_error(tr_y, tr_pred)**0.5)
    print('estop :', mean_squared_error(es_y, es_pred)**0.5)
    if (va_y is not None) and (va_pred is not None):
        print('valid :', mean_squared_error(va_y, va_pred)**0.5)

def rmse_3d(train:pd.DataFrame):
    '''
    3次元rmseの計算
    trainは予測値算出済'''
    errors = list()
    for s in range(4):
        tmp = train.loc[train['sid']==s+1]
        se = np.array([(np.array(tmp[t]) - np.array(tmp[t+'_pred']))**2 for t in config.target_name])
        mse = se.sum(axis=0).mean()
        rmse = mse**0.5
        errors.append(rmse)
    
    errors = np.array(errors)
    rmse = errors.mean()
    return rmse

def makeexportdir():
    now = datetime.datetime.now()
    dirname = 'lgb_' + now.strftime('%m%d_%H:%M:%S')
    dirpath = os.path.join(config.exdir, dirname)
    os.mkdir(dirpath)   # 出力日時記載のフォルダ作成
    return dirpath


def make_submission(test, dirpath):
    '''
    予測済のtest dataframeを投げると投稿ファイルを出力する
    jsonの形式に合わせるためintやlistへの変換を行っている
    予測値の列名は"vel_*_pred"'''
    # testの予測値のためXYをいれかえ
    test['vel_x_pred'] = -1*test['vel_x_pred']
    test['vel_y_pred'] = -1*test['vel_y_pred']

    dic = dict()

    for sub in range(4):
        tmp = test.loc[test['sid']==sub+1]
        sub = 'sub' + str(sub+1)
        dic[sub] = dict()

        for trial in np.unique(np.array(tmp['trial'])):
            dic[sub]['trial'+str(trial)] = [list(a) for a in np.array(tmp.loc[tmp['trial']==trial, [t+'_pred' for t in config.target_name]])]

    json.dump(dic, open(os.path.join(dirpath, 'submission.json'), 'w'))
    print('\nexport succeed')

class base():
    def __init__(self, split_by_subject=False, rand=0):
        self.train = pickle.load(open(config.train_pkl_path, 'rb'))
        self.test = pickle.load(open(config.test_pkl_path, 'rb'))

        self.split_by_subject = split_by_subject
        self.rand = rand

        self.model = None
        self.exornot = None
        self.expath = None

    def train_fn(self):
        # tr_x, tr_y, va_x, va_yを投げるとモデルを返す関数
        pass

    def predict(self):
        # model, xを投げると予測値を返す関数
        pass

    def get_model(self, train, target, index):
        '''
        train、予測対象、インデックスのarrayを投げると予測値を追加したtestを返す
        インデックスは[train, valid, test]の形式
        get_tr_va_indexで得られる形式'''
        x = train.drop(columns=config.drop_list, errors='ignore')
        y = train[target]

        # trainとvalidの分割
        tr_x, tr_y = x[index[0]], y[index[0]]
        es_x, es_y = x[index[1]], y[index[1]]

        # 学習と予測
        model = self.train_fn(tr_x, tr_y, es_x, es_y)
        tr_pred = self.predict(model, tr_x)
        es_pred = self.predict(model, es_x)

        va_y, va_pred = None, None
        # validのインデックスが分割されているとき
        if len(index) == 3:
            va_x, va_y = x[index[2]], y[index[2]]
            va_pred = self.predict(model, va_x)

        print_score(tr_y, tr_pred, es_y, es_pred, va_y, va_pred)
        return model

    def main(self):
        if self.split_by_subject:
            print('split_by_subject :', self.split_by_subject)
    
        train = self.train
        test = self.test
        model = list()

        # x, y, zごとのモデル作成と予測
        for target in config.target_name:
            print('\ntarget :', target)

            # 被験者ごとのモデル作成と予測
            '''if split_by_subject:
                for sid in range(4):
                    sid = sid + 1
                    print('\n被験者id :', sid)
                    train_tmp = train.loc[train['sid']==sid].copy()

                    mod = get_model(train_tmp, target)
                    
                    test.loc[test['sid']==sid, target+'_pred'] = self.predict(mod, test.loc[test['sid']==sid].drop(columns=config.drop_list, errors='ignore'))
                    train.loc[train['sid']==sid, target+'_pred'] = self.predict(mod, train_tmp)
            '''

            index = get_tr_va_index(self.train, rand=self.rand)

            if not self.split_by_subject:
                # 被験者で分割しない場合
                mod = self.get_model(train, target, index)
                test[target+'_pred'] = self.predict(mod, test.drop(columns=config.drop_list, errors='ignore'))
                train[target+'_pred'] = self.predict(mod, train.drop(columns=config.drop_list, errors='ignore'))

                model.append(mod)

        # rmse出力
        tr_rmse = rmse_3d(train[index[0]])
        print('\ntrain rmse :', tr_rmse)
        es_rmse = rmse_3d(train[index[1]])
        print('estop rmse :', es_rmse)

        if len(index)==3:   # validationが設定されているとき
            va_rmse = rmse_3d(train[index[2]])
            print('valid rmse  :', va_rmse)

        #保存
        expath = makeexportdir()
        exornot = input('出力しますか(y/n)')=='y'
        if exornot:
            make_submission(test, expath)

        self.model = model
        self.exornot = exornot
        self.expath = expath
