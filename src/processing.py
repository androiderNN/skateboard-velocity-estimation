from scipy import io as sio
from scipy import signal
import os
import numpy as np

import config

train_raw = sio.loadmat(config.train_path)
# test_raw = sio.loadmat(config.test_path)
# print(test_raw)

def extraction(data_myo, data_vel, data_sta):
    '''
    1trialぶんの筋電位、重心速度、体勢データを入力するとモデルに受け渡し可能な形式に変換する
    
    Parameters
    ----------
    data_myo :ndarray
        (16, 1000)の筋電位データ
    data_vel :ndarray
        (3, 30)の重心速度データ
    data_sta :str
        'goofy' or 'regular'
    '''

    data_myo = np.array([signal.resample(dm, data_vel.shape[1]) for dm in data_myo]).T
    print(data_myo.shape)


def process(data_skater):
    # 不要なarray削除
    data_skater = data_skater[0,0]

    data_myo = data_skater[0]
    data_vel = data_skater[1]
    data_sta = data_skater[2][0]

    for i in range(data_myo.shape[0]):
        extraction(data_myo[i], data_vel[i], data_sta)

process(train_raw['0001'])