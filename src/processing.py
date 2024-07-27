from scipy import io as sio
from scipy import signal
import os
import numpy as np
import pandas as pd

import config

train_raw = sio.loadmat(config.train_path)
test_raw = sio.loadmat(config.test_path)
vel_sample_rate = 30    # 30fps

def extraction(data_myo, isregular):
    '''
    1trial単位の筋電位、重心速度、体勢データを入力すると特徴量を作成し返す
    
    Parameters
    ----------
    data_myo :ndarray
        (16, 1000)の筋電位データ
    data_vel :ndarray
        (3, 30)の重心速度データ
    data_sta :str
        'goofy' or 'regular'
    '''

    data_myo = np.array([signal.resample(dm, vel_sample_rate) for dm in data_myo]).T
    return data_myo

def process(data_skater:np.array, isregular:bool):
    '''
    被験者一人当たりのデータを入力するとモデルに入力可能な形式に変換する
    筋電位データと体勢データを展開'''
    data_myo = data_skater[0,0][0]
    data_extracted = np.array([])

    for arr in data_myo:    # trialごとに分割
        data_extracted = np.append(data_extracted, extraction(arr, isregular))
    
    data_extracted = data_extracted.reshape(16, -1).T   # np.appendでflatになったarrayを2次元に復元
    data_df = pd.DataFrame(data_extracted, columns=config.feature_name)
    data_df['isregular'] = int(isregular)
    return data_df

def make_data(raw_data:np.array, istrain:bool):
    '''
    rawデータを投げるとDataFrameに変換する
    trainデータの場合は速度列も追加して返す'''
    df = pd.DataFrame()

    for i in range(4):
        id = '000' + str(i+1)
        tmp_df = process(data_skater=raw_data[id], isregular=raw_data[id][0,0][-2][0]=='regular')

        # trainならば速度列をDataframeに追加する
        if istrain:
            tmp_df[config.target_name] = raw_data[id][0,0][1].transpose(1,2,0).reshape(3, -1).T
        
        df = pd.concat([df, tmp_df])
    
    df.reset_index(inplace=True, drop=True)
    print(df)
    return df


make_data(train_raw, True)