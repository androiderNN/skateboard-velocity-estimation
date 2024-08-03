import numpy as np
import pandas as pd

feature_name = ['TA R', 'TA L', 'LG R', 'LG L', 'RF R', 'RF L', 'VL R', 'VL L', 'ST R', 'ST L', 'GM R', 'GM L', 'EM R', 'EM L', 'DE R', 'DE L']
n = 1000
dt = 1/2000

def fft(data_myo):
    '''
    一人分の筋電位データを入力すると変換後の特徴量dfを出力する'''
    overlap = 32
    winf = np.hanning(2*overlap)    # 窓関数
    acf = 1/(sum(winf)/2*overlap)   # 窓関数の補正値
    freq = np.fft.fftfreq(2*overlap, d=dt)[:overlap]    # 周波数

    fft = np.zeros([data_myo.shape[0], 30, data_myo.shape[1], overlap], dtype=float)
    
    for trial in range(data_myo.shape[0]):
        for col in range(data_myo.shape[1]):
            for timepoint in range(29):
                myo = data_myo[trial, col, round(1000*(timepoint+1)/30-overlap):round(1000*(timepoint+1)/30+overlap)]
                fft[trial, timepoint, col] = abs(np.fft.fft(myo*winf)/overlap)[:overlap]*acf

            fft[trial, :, col, -1] = fft[trial, :, col, -2] # timepoint=30のときはtp=29で埋める
    
    fft = fft[:, :, :, 1:]  # 0は0Hzなので除く
    freq = freq[1:]

    fft_reshaped = fft.reshape(fft.shape[0], fft.shape[1], -1)   # colとfreqを同一次元に
    fft_reshaped = fft_reshaped.reshape(-1, fft_reshaped.shape[-1])    # trialとtimepointを同一次元に
    col = [c+'_fft'+str(f) for c in feature_name for f in freq]

    fft_df = pd.DataFrame(fft_reshaped, columns=col)
    fft_df[['trial', 'timepoint']] = [[tr, ti] for tr in range(fft.shape[0]) for ti in range(fft.shape[1])]

    return fft_df
