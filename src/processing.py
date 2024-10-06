import os, pickle
import numpy as np
import pandas as pd
from scipy import io as sio

import config
from features import iemg, fft, process_core
from models import clustering

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

def process_trial(data_sub:np.array, isregular:bool, sid:int, ie):
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
    # if cluster_model:
    #     clt_df = cluster_model.make_df(ie)
    #     data_df = pd.merge(data_df, clt_df, on='trial')

    return data_df

def process_timepoint(data_sub, fft_df):
    '''
    被験者一人当たりのデータを入力するとtimepointに固有な特徴量を作成する'''
    data_myo = data_sub[0,0][0]  # 筋電位データ
    num_trial = data_myo.shape[0]
    data_df = pd.DataFrame([[t+1, i] for t in range(num_trial) for i in range(30)], columns=['trial', 'timepoint'])

    data_df = pd.merge(data_df, fft_df, on=['trial', 'timepoint'])
    return data_df

def make_data(raw_data:np.array, istrain:bool, fft_df, ie):
    '''
    rawデータを投げるとDataFrameに変換する
    trainデータの場合は速度列も追加して返す'''
    trial_df = pd.DataFrame()
    timepoint_df = pd.DataFrame()

    for i in range(4):  # 被験者ごとにprocessに投げる
        sid = '000' + str(i+1)
        tmp_df = process_trial(raw_data[sid], raw_data[sid][0,0][-2][0]=='regular', i+1, ie[i])
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
    redump_fft = False
    redump_iemg = False
    compress_fft = False

    print(f'redump_fft = {redump_fft}')
    print(f'redump_iemg = {redump_iemg}')
    print(f'compress_fft = {compress_fft}')

    # fft_dfとiemgの再作成
    if redump_fft:
        print('\nmaking fft_df')
        fft.dump_fft_df()
    
    if redump_iemg:
        print('making iemg data')
        iemg.dump_iemg()
    
    # train data
    train_raw = sio.loadmat(config.train_raw_path)
    fft_df = pickle.load(open(config.fft_train_path, 'rb'))
    ie = pickle.load(open(config.iemg_train_path, 'rb'))

    tr_trial_df, tr_timepoint_df = make_data(train_raw, True, fft_df, ie)
    
    # test data
    test_raw = sio.loadmat(config.test_raw_path)
    fft_df = pickle.load(open(config.fft_test_path, 'rb'))
    ie = pickle.load(open(config.iemg_test_path, 'rb'))

    te_trial_df, te_timepoint_df = make_data(test_raw, False, fft_df, ie)

    if compress_fft:
        cols = [c for c in tr_timepoint_df.columns if c[4:8]=='_fft' and c[-4:]!='dens']
        process_core.compress(tr_timepoint_df, te_timepoint_df, cols, n_components=100)

    # dataframe保存
    pickle.dump(tr_trial_df, open(config.train_trial_path, 'wb'))
    pickle.dump(te_trial_df, open(config.test_trial_path, 'wb'))
    pickle.dump(tr_timepoint_df, open(config.train_path, 'wb'))
    pickle.dump(te_timepoint_df, open(config.test_path, 'wb'))

    print('\nsuccessfully finished.')
