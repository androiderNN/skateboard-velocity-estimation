import os, pickle, datetime, json, sys, math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'features'))
import process_core

train = pickle.load(open(config.train_path, 'rb'))
test = pickle.load(open(config.test_path, 'rb'))

def get_tr_va_index(train, es_size=0.1, va_size=0, rand=0):
    '''
    trainのデータフレームを投げるとtrain, estop, validのindexを作成'''
    if es_size+va_size>1:
        raise ValueError
    
    sid = train['sid'].tolist()
    trial = train['trial'].tolist()
    train = pd.concat([train, pd.DataFrame([str(sid[i])+'_'+str(trial[i]) for i in range(len(train))], columns=['trial_id'])], axis=1)    # sidとtrialでtrial_idを作成
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

    # '''
    # trial%10==0のデータをバリデーションに使用'''
    # tr_index = train['trial']%10!=2
    # es_index = train['trial']%10==2

    # return [tr_index, es_index]

def get_kfold_index(train, n_fold=4, rand=0):
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

def rmse(y, y_pred):
    return mean_squared_error(y, y_pred)**0.5

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

def smoothing(df, ksize=5):
    df = df.reindex(columns=['sid', 'trial', 'timepoint', 'vel_x', 'vel_y', 'vel_z', 'vel_x_pred', 'vel_y_pred', 'vel_z_pred'])
    array = np.array(df)
    for sid in range(1,5):
        for trial in np.unique(array[array[:,0]==sid, 1]):
            for target in range(-3,0):
                t = array[(array[:,0]==sid)&(array[:,1]==trial), target]

                if ksize == 3:
                    t = [t[0]] + [t[i:i+3].mean() for i in range(28)] + [t[29]]
                elif ksize == 5:
                    t = list(t[:2]) + [t[i:i+5].mean() for i in range(26)] + list(t[28:])
                else:
                    raise ValueError
                
                array[(array[:,0]==sid)&(array[:,1]==trial), target] = t

    df = pd.DataFrame(array, columns=df.columns)
    df = df.astype({'sid': 'int16', 'trial': 'int16', 'timepoint': 'int16'})
    return df

def makeexportdir(type:str):
    now = datetime.datetime.now()
    dirname = type + '_' + now.strftime('%m%d_%H:%M:%S')
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

class modeler_base():
    '''
    各機械学習アルゴリズムの最も単純なラッパ'''
    def __init__(self, modeler_params, rand):
        self.model = None
        self.params = None
        self.rand = rand

    def train(self, tr_x, tr_y, va_x, va_y):
        '''
        モデルを学習する関数'''
        pass

    def predict(self, x):
        '''
        xを渡すと予測値を返す関数'''
        pass

class holdout_training():
    def __init__(self, modeler, params, score_fn):
        '''
        hold-outで学習・予測・スコア算出を行うクラス
        modeler : modeler_baseを継承したクラス
        score_fn : y, y_predを入力するとスカラーのスコアを返す関数'''
        self.modeler = modeler(params=params['modeler_params'], rand=params['rand'])
        self.params = params
        self.score_fn = score_fn

    def main(self, train, col, target, test, index=None):
        '''
        関数内でインデックス分割、学習、スコア出力、testデータの予測出力まで行う'''
        if self.params['normalize']:
            cols = train.drop(columns=config.drop_list, errors='ignore').columns.tolist()
            train, test = process_core.normalize(train, test, cols)
        
        # データ分割
        index = get_tr_va_index(train, rand=self.params['rand']) if index is None else index
        tr_x, tr_y = train.loc[index[0], col], train.loc[index[0], target]
        es_x, es_y = train.loc[index[1], col], train.loc[index[1], target]

        # 学習
        self.modeler.train(tr_x, tr_y, es_x, es_y)

        # スコア出力
        tr_pred = self.modeler.predict(tr_x)
        tr_score = self.score_fn(tr_y, tr_pred)
        print('train score :', tr_score)
        es_pred = self.modeler.predict(es_x)
        es_score = self.score_fn(es_y, es_pred)
        print('estop score :', es_score)

        # 予測
        train_pred = self.modeler.predict(train[col])
        test_pred = self.modeler.predict(test[col])
        return train_pred, test_pred

class cv_training():
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

    def main(self, train, col, target, test, index=None):
        '''
        trainデータ、特徴量の列名リスト、ターゲットの列名、テストデータを投げると学習と結果出力を行いtestデータの予測値を返す'''
        if self.params['normalize']:
            cols = train.drop(columns=config.drop_list, errors='ignore').columns.tolist()
            train, test = process_core.normalize(train, test, cols)

        # valid分割
        index = get_tr_va_index(train, rand=self.params['rand']) if index is None else index
        tr_idx, va_idx = index
        tr = train[tr_idx].copy()  # 学習用データ
        va = train[va_idx].copy()  # バリデーション用データ

        # fold分割
        index_array = get_kfold_index(tr, n_fold=4, rand=self.params['rand'])

        for i, index in enumerate(index_array):
            print('\nFold', i+1)
            tr_x, tr_y = tr.loc[index[0], col], tr.loc[index[0], target]    # fold内での学習用データ
            es_x, es_y = tr.loc[index[1], col], tr.loc[index[1], target]    # fold内でのearly stopping用データ

            # 学習
            modeler = self.mdr(params=self.params['modeler_params'], rand=self.params['rand'])
            modeler.train(tr_x, tr_y, es_x, es_y)

            # スコア出力
            tr_pred = modeler.predict(tr_x)
            tr_score = self.score_fn(tr_y, tr_pred)
            print('train score :', tr_score)

            es_pred = modeler.predict(es_x)
            es_score = self.score_fn(es_y, es_pred)
            print('estop score :', es_score)

            self.modelers.append(modeler)

        # スコア出力
        print('\nmean prediction')
        train_pred = self.predict(tr[col])
        train_score = self.score_fn(tr[target], train_pred)
        print('train score :', train_score)

        valid_pred = self.predict(va[col])
        valid_score = self.score_fn(va[target], valid_pred)
        print('valid score :', valid_score, '\n')

        # testデータの予測
        train_pred = self.predict(train[col])
        test_pred = self.predict(test[col])
        return train_pred, test_pred

class vel_prediction():
    def __init__(self, modeler, params):
        self.params = params

        self.col = [c for c in train.columns if c not in config.drop_list]
        self.modeler = modeler
        self.expath = None
        self.exornot = False

        self.trainer_array = list()
        self.train_pred = train.loc[:, ['sid', 'trial', 'timepoint', 'vel_x', 'vel_y', 'vel_z']]
        self.test_pred = test.loc[:, ['sid', 'trial', 'timepoint']]

    def main(self):
        if self.params['split_by_subject']:
            print('split_by_subject :', self.split_by_subject)
        
        # trainerの定義
        if self.params['use_cv']: # cross validation
            trainer_class = cv_training
        else:   # hold out
            trainer_class = holdout_training

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

            if not self.params['split_by_subject']:   # 被験者で分割しない場合
                # trainerの定義 cv_trainerまたはholdout_trainer
                trainer = trainer_class(self.modeler, self.params, score_fn=rmse)

                # 学習・予測
                tr_pred, te_pred = trainer.main(train, self.col, target, test)
                self.train_pred[target+'_pred'] = tr_pred
                self.test_pred[target+'_pred'] = te_pred

                self.trainer_array.append(trainer)

        # 予測値の平滑化
        if self.params['smoothing']:
            self.train_pred = smoothing(self.train_pred, ksize=5)
            self.test_pred = smoothing(self.test_pred, ksize=5)

        # rmse出力 cvかhoかでvalidデータの使用目的（estop/valid)が異なるため注意
        index = get_tr_va_index(train, rand=self.params['rand'])
        tr_rmse = rmse_3d(self.train_pred[index[0]])
        print('\ntrain rmse :', tr_rmse)
        es_rmse = rmse_3d(self.train_pred[index[1]])
        print('validation rmse :', es_rmse)

        #保存
        self.expath = makeexportdir(self.params['modeltype'])
        if self.params['verbose']:
            self.exornot = input('\n出力しますか(y/n)')=='y'
        if self.exornot:
            os.mkdir(self.expath)   # 出力日時記載のフォルダ作成
            make_submission(self.test_pred, self.expath)
            pickle.dump(self.train_pred, open(os.path.join(self.expath, 'train_pred.pkl'), 'wb'))
            pickle.dump(self.test_pred, open(os.path.join(self.expath, 'testpred.pkl'), 'wb'))
            pickle.dump(self.params, open(os.path.join(self.expath, 'params.pkl'), 'wb'))

default_params = {
    'modeltype': None,               # 'lgb'など
    'rand': 0,                  # シード
    'use_cv': False,            # cross validationの使用可否
    'verbose': True,            # 出力の可否
    'normalize': False,         # データの標準化可否
    'smoothing': True,          # 予測値の平滑化可否
    'split_by_subject': False,
    'modeler_params': None      # modelerに渡すパラメータ
}
