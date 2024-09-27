import os, pickle
import numpy as np
import pandas as pd
from scipy import io as sio
from scipy import signal

import config
from features import iemg, fft, process_core
from models import clustering

train_raw = pickle.load(open(config.train_raw_path, 'rb'))
test_raw = pickle.load(open(config.test_raw_path, 'rb'))

def convert_y(df, goofy_only:bool):
    '''
    データを左右対称に入れ替える関数'''
    if goofy_only:
        df_tmp = df.loc[df['isregular']==0]
    else:
        df_tmp = df

    # 列名の変更により筋電位データの左右入れ替え
    col = [c[:-1]+c[-1].translate(str.maketrans({'L':'R', 'R':'L'})) for c in df_tmp.columns]
    df_tmp.columns = col
    
    # 速度の実測値・予測値の正負を入れ替え
    if 'vel_y' in col:
        df_tmp.loc[:, 'vel_y'] = df_tmp['vel_y']*-1
    if 'vel_y_pred' in col:
        df_tmp.loc[:, 'vel_y_pred'] = df_tmp['vel_y_pred']*-1
    
    if goofy_only:
        df.loc[df['isregular']==0] = df_tmp
    else:
        df = df_tmp
    
    return df

def process_trial(data_sub:np.array, isregular:bool, sid:int, ie, cluster_model):
    '''
    被験者一人当たりのデータを入力するとtrialに固有な特徴量を作成する
    timepoint固有の特徴量はprocess_timepoint関数で作成'''
    data_myo = data_sub[0,0][0]  # 筋電位データ
    num_trial = data_myo.shape[0]

    # 基本列作成
    data_df = pd.DataFrame([t+1 for t in range(num_trial)], columns=['trial'])
    data_df['isregular'] = int(isregular)
    data_df.insert(0, 'sid', sid)

    # iemg計算
    iemg_df = iemg.iemg(data_myo, ie=ie)
    data_df = pd.merge(data_df, iemg_df, on='trial')

    # iemgクラスタリング
    # clt_df = cluster_model.make_df(ie)
    # data_df = pd.merge(data_df, clt_df, on='trial')

    # fft計算
    # fft_array = fft.fft_core(data_myo)[:,:,:31]
    # fft_array = fft_array.reshape(fft_array.shape[0], -1)
    # freq = fft.get_freq(data_myo)[:31]
    # fft_df = pd.DataFrame(fft_array, columns=[c+str(f) for c in config.feature_name for f in freq])
    # fft_df['trial'] = [t+1 for t in range(num_trial)]
    # data_df = pd.merge(data_df, fft_df, on='trial')
    
    # sidのonehot encoding
    # sid, cl = process_core.onehot(data_df['sid'])
    # cl = ['sid_'+i for i in cl]
    # # data_df[cl] = pd.DataFrame(sid)
    # print(sid.shape)
    # data_df[cl] = sid

    return data_df

def process_timepoint(data_sub, fft_df):
    '''
    被験者一人当たりのデータを入力するとtimepointに固有な特徴量を作成する'''
    data_myo = data_sub[0,0][0]  # 筋電位データ
    num_trial = data_myo.shape[0]
    data_df = pd.DataFrame([[t+1, i] for t in range(num_trial) for i in range(30)], columns=['trial', 'timepoint'])

    # fft計算
    if fft_df is None:  # fft_dfが与えられていない時は再度作成
        fft_df = fft.fft_onVelosityTime(data_myo)
    data_df = pd.merge(data_df, fft_df, on=['trial', 'timepoint'])

    return data_df

def make_data(raw_data:np.array, istrain:bool, fft_df, ie, cluster_model):
    '''
    rawデータを投げるとDataFrameに変換する
    trainデータの場合は速度列も追加して返す'''
    trial_df = pd.DataFrame()
    timepoint_df = pd.DataFrame()

    for i in range(4):  # 被験者ごとにprocessに投げる
        sid = '000' + str(i+1)
        # tmp_df = process(raw_data[sid], raw_data[sid][0,0][-2][0]=='regular', i+1, fft_df[i], ie[i], cluster_model)
        tmp_df = process_trial(raw_data[sid], raw_data[sid][0,0][-2][0]=='regular', i+1, ie[i], cluster_model)
        trial_df = pd.concat([trial_df, tmp_df])

        tmp_df = pd.merge(process_timepoint(raw_data[sid], fft_df[i]), tmp_df, on='trial')
        # trainならば速度列をDataframeに追加する
        if istrain:
            tmp_df[config.target_name] = raw_data[sid][0,0][1].reshape(-1,3,30).transpose(0,2,1).reshape(-1,3)
        
        timepoint_df = pd.concat([timepoint_df, tmp_df])
    
    # sidのonehot encoding
    sid, cl = process_core.onehot(timepoint_df['sid'])
    cl = ['sid_'+i for i in cl]
    timepoint_df[cl] = sid
    
    sid, cl = process_core.onehot(trial_df['sid'])
    cl = ['sid_'+i for i in cl]
    trial_df[cl] = sid

    trial_df.reset_index(inplace=True, drop=True)
    timepoint_df.reset_index(inplace=True, drop=True)
    return trial_df, timepoint_df

if __name__ == '__main__':
    fft_df = pickle.load(open(config.fft_train_path, 'rb'))
    ie = pickle.load(open(config.iemg_train_path, 'rb'))

    iemg_cluster_model = clustering.clustering_trial(num_pick=10, n_clusters=3)
    iemg_cluster_model.fit(ie)

    tr_trial_df, tr_timepoint_df = make_data(train_raw, True, fft_df, ie, iemg_cluster_model)
    
    fft_df = pickle.load(open(config.fft_test_path, 'rb'))
    ie = pickle.load(open(config.iemg_test_path, 'rb'))
    te_trial_df, te_timepoint_df = make_data(test_raw, False, fft_df, ie, iemg_cluster_model)

    cols = [c for c in tr_timepoint_df.columns if c[4:8]=='_fft' and c[-4:]!='dens']
    tr_timepoint_df, te_timepoint_df = process_core.compress(tr_timepoint_df, te_timepoint_df, cols, n_components=100)

    pickle.dump(tr_trial_df, open(config.train_trial_path, 'wb'))
    pickle.dump(te_trial_df, open(config.test_trial_path, 'wb'))
    pickle.dump(tr_timepoint_df, open(config.train_path, 'wb'))
    pickle.dump(te_timepoint_df, open(config.test_path, 'wb'))

    print('successfully finished.')
