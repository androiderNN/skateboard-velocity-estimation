import numpy as np
import pandas as pd

feature_name = ['TA R', 'TA L', 'LG R', 'LG L', 'RF R', 'RF L', 'VL R', 'VL L', 'ST R', 'ST L', 'GM R', 'GM L', 'EM R', 'EM L', 'DE R', 'DE L']
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
    col = [c+'_fft'+str(f) for c in feature_name for f in freq]

    fft_df = pd.DataFrame(fft_reshaped, columns=col)
    fft_df[['trial', 'timepoint']] = [[tr+1, ti] for tr in range(fft.shape[0]) for ti in range(fft.shape[1])]

    return fft_df
