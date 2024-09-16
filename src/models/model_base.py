import os, pickle, datetime, json, sys, math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

train = pickle.load(open(config.train_pkl_path, 'rb'))
test = pickle.load(open(config.test_pkl_path, 'rb'))

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

def get_kfold_index(train, n_fold=4):
    '''
    trainのデータフレームを投げると[train, estop]*n_foldのindexを作成'''
    sid = train['sid'].tolist()
    trial = train['trial'].tolist()
    train['trial_id'] = [str(sid[i])+'_'+str(trial[i]) for i in range(len(train))]    # sidとtrialでtrial_idを作成
    trial_id = np.unique(np.array(train['trial_id']))

    num_trial_id = len(trial_id)
    tmp = [0 for _ in range(num_trial_id)]
    n_ids = math.floor(num_trial_id/n_fold)

    # tmpは0~n_fold-1の整数がランダムにtrial_idの数並んだ配列
    for i in range(n_fold-1):
        tmp[(i+1)*n_ids:(i+2)*n_ids] = [i+1]*n_ids
    tmp = np.array(tmp)
    np.random.shuffle(tmp)

    va_id_split = [[id for j, id in enumerate(trial_id) if i==tmp[j]] for i in range(n_fold)]    # [[va_ids_0],[va_ids_1]...]
    
    index_array = list()    # [[tr_index, va_index]]*n_fold
    for va_id in va_id_split:
        tr_index = [id not in va_id for id in train['trial_id']]
        va_index = [id in va_id for id in train['trial_id']]
        index_array.append([tr_index, va_index])

    train.drop(columns='trial_id', inplace=True)    # trainは参照なのでtrial_idを削除しておく

    return index_array

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
    def __init__(self, split_by_subject, rand, index, verbose=True):
        self.train = train.copy()
        self.test = test.copy()

        self.split_by_subject = split_by_subject
        self.rand = rand
        self.verbose = verbose

        if index is None:
            self.index = get_tr_va_index(self.train, rand=self.rand)
        else:
            self.index = index

        self.model = None
        self.expath = None
        self.exornot = False

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

    def get_prediction(self):
        '''
        self.train, self.testから予測値の列を返す
        主にKFold用'''
        cols = [t+'_pred' for t in config.target_name]
        tr_pred = self.train_pred[cols]
        te_pred = self.test_pred[cols]
        return tr_pred, te_pred

    def main(self):
        if self.split_by_subject:
            print('split_by_subject :', self.split_by_subject)
    
        train = self.train.copy()
        test = self.test.copy()
        self.model = list()

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

            if not self.split_by_subject:
                # 被験者で分割しない場合
                mod = self.get_model(train, target, self.index)
                test[target+'_pred'] = self.predict(mod, test.drop(columns=config.drop_list, errors='ignore'))
                train[target+'_pred'] = self.predict(mod, train.drop(columns=config.drop_list, errors='ignore'))

                self.model.append(mod)

        # rmse出力
        tr_rmse = rmse_3d(train[self.index[0]])
        print('\ntrain rmse :', tr_rmse)
        es_rmse = rmse_3d(train[self.index[1]])
        print('estop rmse :', es_rmse)

        if len(self.index)==3:   # validationが設定されているとき
            va_rmse = rmse_3d(train[self.index[2]])
            print('valid rmse  :', va_rmse)

        #保存
        self.expath = makeexportdir()
        if self.verbose:
            self.exornot = input('出力しますか(y/n)')=='y'
        if self.exornot:
            os.mkdir(self.expath)   # 出力日時記載のフォルダ作成
            make_submission(test, self.expath)

        self.train_pred = train[[c+'_pred' for c in config.target_name]]
        self.test_pred = test[[c+'_pred' for c in config.target_name]]

def train_CV(modeler_class, n_fold, rand):
    tr = train.copy()
    te = test.copy()
    tr_pred, te_pred = list(), list()

    index = get_kfold_index(tr)  # 二次元配列

    for f in range(n_fold):
        print('\nFold', f)
        ins = modeler_class(split_by_subject=False, rand=rand, index=index[f], verbose=False)
        ins.main()
        pred = ins.get_prediction()
        tr_pred.append(pred[0])
        te_pred.append(pred[1])
    
    cols = [t+'_pred' for t in config.target_name]
    tr[cols] = np.array(tr_pred).mean(axis=0)
    te[cols] = np.array(te_pred).mean(axis=0)
    print(tr)
    
    rmse = rmse_3d(tr)
    print('\n\ntrain rmse:', rmse)

    #保存
    exornot = input('出力しますか(y/n)')=='y'
    if exornot:
        expath = makeexportdir()
        os.mkdir(expath)   # 出力日時記載のフォルダ作成
        make_submission(te, expath)
