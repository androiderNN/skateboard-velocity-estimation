import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from features import process_core

def clustering_kmeans(x, n_clusters=5):
    '''
    k-最近傍法でクラスタリング'''
    km = KMeans(n_clusters=n_clusters)
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

class clustering():
    def __init__(self, num_pick=10, n_clusters=5):
        self.clt_fn = clustering_kmeans
        self.clt_model = None   # クラスタリングを行うモデル

        self.num_pick = num_pick
        self.n_clusters = n_clusters

    def pick(self, ie, num_pick, sid=None):
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

    def convert(self, ie_array):
        '''
        4人分のiemgデータを処理して連結'''
        df = pd.DataFrame()

        for i in range(4):
            tmp, col = self.pick(ie_array[i], self.num_pick, sid=i+1)
            df = pd.concat([df, tmp])
        
        return df, col

    def fit(self, ie_array):
        tmp_df, col = self.convert(ie_array)
        self.clt_model, _ = self.clt_fn(tmp_df[col], self.n_clusters)
    
    def make_df(self, ie):
        '''
        一人分のieを入力するとクラスタリング後のラベルdfを出力する'''
        tmp_df, col = self.pick(ie, self.num_pick)
        clt = self.clt_model.predict(tmp_df[col])
        tmp_df[['clt_'+str(i) for i in range(self.n_clusters)]] = process_core.onehot(clt)[0]
        tmp_df.drop(columns=col, inplace=True)  # trial, col, sid, クラスタリング後のonehotラベル列　(16*trials,n_clusters+3) 

        array = np.array(tmp_df.iloc[:, 2:])   # (16*trials, n_clusters)
        array = array.reshape(tmp_df['trial'].max(), -1) # (trial, 16*n_clusters)

        clt_df = pd.DataFrame(array, columns=[c+'_'+tc for c in config.feature_name for tc in tmp_df.columns[2:]])
        clt_df['trial'] = [i+1 for i in range(len(clt_df))]

        return clt_df
