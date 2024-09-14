import os, pickle, datetime, json, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

rand = 1
split_by_subject = False
modeltype = 'lgb'

def lgb_train(tr_x, tr_y, va_x, va_y):
    '''
    trainとvalidのDataframeを投げるとlightgbmのモデルを返す'''
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'random_state': rand,
        'verbose': -1,
        # 'reg_alpha': 1,
        # 'reg_lambda': 1,
        # 'min_child_samples': 100
    }
    
    tr_lgb = lgb.Dataset(tr_x, tr_y)
    va_lgb = lgb.Dataset(va_x, va_y)

    model = lgb.train(
        params=params,
        train_set=tr_lgb,
        num_boost_round=1000,
        valid_sets=va_lgb,
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=3, verbose=True)]
    )

    return model

def lgb_predict(model, x:pd.DataFrame):
    return model.predict(x.drop(columns=config.drop_list, errors='ignore'))

def print_score(tr_y, tr_pred, es_y, es_pred, va_y, va_pred):
    '''
    正解と予測値を渡すとスコアを出力する'''
    print('train :', mean_squared_error(tr_y, tr_pred)**0.5)
    print('estop :', mean_squared_error(es_y, es_pred)**0.5)
    if (va_y is not None) and (va_pred is not None):
        print('valid :', mean_squared_error(va_y, va_pred)**0.5)

def make_submission(test):
    '''
    予測済のtest dataframeを投げると投稿ファイルを出力する
    jsonの形式に合わせるためintやlistへの変換を行っている'''
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

    now = datetime.datetime.now()
    dirname = 'lgb_' + now.strftime('%m%d_%H:%M:%S')
    dirpath = os.path.join(config.exdir, dirname)
    os.mkdir(dirpath)   # 出力日時記載のフォルダ作成
    json.dump(dic, open(os.path.join(dirpath, 'submission.json'), 'w'))

    print('\nexport succeed')
    return dirpath

def get_tr_va_index(train, es_size=0.2, va_size=0):
    '''
    trainのデータフレームを投げるとtrain, val, va_esのindexを作成'''
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

def get_model(train, target, index):
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
    model = lgb_train(tr_x, tr_y, es_x, es_y)
    tr_pred = lgb_predict(model, tr_x)
    es_pred = lgb_predict(model, es_x)

    va_y, va_pred = None, None
    # validのインデックスが分割されているとき
    if len(index) == 3:
        va_x, va_y = x[index[2]], y[index[2]]
        va_pred = lgb_predict(model, va_x)

    print_score(tr_y, tr_pred, es_y, es_pred, va_y, va_pred)
    return model

def rmse_3d(train:pd.DataFrame):
    '''
    3次元rmseの計算
    trainは予測値算出済'''
    se = np.array([(np.array(train[t]) - np.array(train[t+'_pred']))**2 for t in config.target_name])
    mse = np.sum(se) / se.shape[1]
    rmse = mse**0.5
    return rmse

def main():
    print('split_by_subject :', split_by_subject)

    train = pickle.load(open(config.train_pkl_path, 'rb'))
    test = pickle.load(open(config.test_pkl_path, 'rb'))
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
                
                test.loc[test['sid']==sid, target+'_pred'] = lgb_predict(mod, test.loc[test['sid']==sid].drop(columns=config.drop_list, errors='ignore'))
                train.loc[train['sid']==sid, target+'_pred'] = lgb_predict(mod, train_tmp)
        '''

        index = get_tr_va_index(train)

        if not split_by_subject:
            # 被験者で分割しない場合
            mod = get_model(train, target, index)
            test[target+'_pred'] = lgb_predict(mod, test.drop(columns=config.drop_list, errors='ignore'))
            train[target+'_pred'] = lgb_predict(mod, train.drop(columns=config.drop_list, errors='ignore'))

            model.append(mod)

    # rmse出力
    tr_rmse = rmse_3d(train[index[0]])
    print('\ntrain rmse :', tr_rmse)
    es_rmse = rmse_3d(train[index[1]])
    print('estop rmse :', es_rmse)

    if len(index)==3:   # validationが設定されているとき
        va_rmse = rmse_3d(train[index[2]])
        print('valid rmse  :', va_rmse)

    # feature importance出力
    importance_df = pd.DataFrame( \
        {t: model[i].feature_importance(importance_type='gain') for i, t in enumerate(config.target_name)}, \
        index=train.drop(columns=config.drop_list, errors='ignore').columns, \
        columns=config.target_name)
    importance_df['mean'] = importance_df.mean(axis=1)
    # importance_df.sort_values('mean', ascending=False, inplace=True)
    print(importance_df.head(10), '\n')

    i = input('出力しますか(y/n)')=='y'
    if i:
        expath = make_submission(test)
        importance_df.to_csv(os.path.join(expath, 'importance.csv'))

if __name__=='__main__':
    main()
