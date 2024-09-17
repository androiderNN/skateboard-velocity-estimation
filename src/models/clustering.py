import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from features import iemg

def clustering_kmeans(x, n_cluster=5):
    '''
    k-最近傍法でクラスタリング'''
    km = KMeans(n_clusters=n_cluster)
    km.fit(x)
    clt = km.predict(x)
    return km, clt

def plot_cluster(x, cluster_id):
    '''
    クラスタの図示'''
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    x_compressed = tsne.fit_transform(x)

    for c in np.unique(np.array(cluster_id)):
        index = cluster_id==c
        plt.scatter(x_compressed[index, 0], x_compressed[index, 1])
    
    plt.show()

def calc_dist(x):
    '''
    n_clustersに応じた重心までの距離の平方和を描画する
    n_clusterの最適化に'''
    dist = list()
    K = range(1,10)

    for k in K:
        km, _ = clustering_kmeans(x, k)
        dist.append(sum(np.min(cdist(x, km.cluster_centers_, 'euclidean'), axis=1)) / x.shape[0])
    
    plt.plot(K, dist, 'bx-')
    plt.xlabel('n_cluster')
    plt.ylabel('distortion')
    plt.show()

def pick(ie, num_pick, sid=None):
    '''
    1000点から均等にnum_pickを抽出してdataframeに変換する
    dataframeとカラム名を返す'''
    # 抽出
    num_trial = ie.shape[0]
    index = [0] + [round((i+1)*ie.shape[2]/(num_pick-1))-1 for i in range(num_pick-1)]
    ie = ie[:,:,index]

    # dataframeに変換
    ie = ie.reshape(-1, num_pick)
    col = ['iemg_pick_'+str(i) for i in range(num_pick)]
    ie_df = pd.DataFrame(ie, columns=col)
    ie_df['trial'] = [t+1 for t in range(num_trial) for _ in range(16)]
    ie_df['col'] = [c for _ in range(num_trial) for c in config.feature_name]

    if sid is not None:
        ie_df['sid'] = sid

    return ie_df, col

def convert(ie, num_pick):
    '''
    4人分のiemgデータを処理して連結'''
    df = pd.DataFrame()

    for i in range(4):
        tmp, col = pick(ie[i], num_pick, sid=i+1)
        df = pd.concat([df, tmp])
    
    return df, col

def clustering(data_myo, ie, num_pick=10, n_clusters=5):
    num_trial = data_myo.shape[0]
    df = pd.DataFrame([[t+1, i] for t in range(num_trial) for i in range(30)], columns=['trial', 'timepoint'])

    tmp_df, col = convert(ie, num_pick=num_pick)
    km, clt = clustering_kmeans(tmp_df[col], n_clusters)
