import numpy as np

def onehot(x):
    '''
    one-hot encodigする関数
    一列分のデータを入力するとデータ数*クラス数のndarrayを返す'''
    x = np.array(x)
    cl = np.unique(x)
    array = np.zeros(shape=(x.shape[0], cl.shape[0]))

    for i, c in enumerate(cl):
        array[x==c, i] = 1

    cl = [str(c) for c in cl]
    return array, cl
