import os, sys, pickle
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
import process_core
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

dt = 1/2000

def fft_core(data):
    '''
    筋電位データのndarrayを投げると変換後のndarrayと周波数のarrayを返す
    入力データの形式は[trial, features, timepoint]'''
    n = data.shape[2]   # サンプル数
    windowF = np.hanning(n) # 窓関数
    acf = n/sum(windowF)    # 窓関数の補正値

    data = data*windowF
   
    fft = np.zeros([data.shape[0], data.shape[1], n//2])

    for trial in range(data.shape[0]):
        for feature in range(data.shape[1]):
            fft[trial, feature, :] = abs(np.fft.fft(data[trial, feature, :])[:n//2])
    
    fft /= n//2 # サンプル数で正規化
    fft *= acf  # 窓関数の補正

    # 0は0Hzなので除く
    fft = fft[:,:,1:]

    return fft

def get_freq(data):
    n = data.shape[2]
    freq = np.fft.fftfreq(data.shape[2], d=dt)[1:n//2]   # 0は0Hzなので除く
    return freq

def fft_onVelosityTime(data_myo):
    '''
    一人分の筋電位データを入力すると変換後の特徴量dfを出力する'''
    # 高速フーリエ変換
    overlap = 32
    fft = np.zeros([data_myo.shape[0], 30, data_myo.shape[1], overlap-1], dtype=float)

    data_myo /= data_myo.max(axis=2)[:,:,np.newaxis]

    for timepoint in range(29):
        myo = data_myo[:, :, round(1000*(timepoint+1)/30-overlap):round(1000*(timepoint+1)/30+overlap)] # 筋電位データ切り出し
        f = fft_core(myo)
        fft[:, timepoint, :, :31] = f
    
    fft[:,29,:,:] = fft[:,28,:,:]   # timepoint=30のときはtp=29で埋める

    freq = get_freq(myo)    # 周波数

    # 周波数のndarrayをDataFrameに変換
    fft_reshaped = fft.reshape(fft.shape[0], fft.shape[1], -1)   # colとfreqを同一次元に
    fft_reshaped = fft_reshaped.reshape(-1, fft_reshaped.shape[-1])    # trialとtimepointを同一次元に
    fft_col = [c+'_fft'+str(f) for c in config.feature_name for f in freq]

    fft_df = pd.DataFrame(fft_reshaped, columns=fft_col)

    # 各colのfft平均を挿入
    fft_df[['fft_dens_'+col for col in config.feature_name]] = fft.mean(axis=3).reshape(-1, fft.shape[2])

    # 次元圧縮
    fft_df = process_core.compress(fft_df, fft_col, 100, 'fft')

    fft_df[['trial', 'timepoint']] = [[tr+1, ti] for tr in range(fft.shape[0]) for ti in range(fft.shape[1])]

    return fft_df

def dump_fft_df():
    train = pickle.load(open(config.train_path, 'rb'))
    test = pickle.load(open(config.test_path, 'rb'))

    l = list()
    for i in range(4):
        l.append(fft_onVelosityTime(train['000'+str(i+1)][0,0][0]))
        
    pickle.dump(l, open(config.fft_train_path, 'wb'))

    l = list()
    for i in range(4):
        l.append(fft_onVelosityTime(test['000'+str(i+1)][0,0][0]))
        
    pickle.dump(l, open(config.fft_test_path, 'wb'))
