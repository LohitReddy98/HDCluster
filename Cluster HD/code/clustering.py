import os
from random import randint
import sys
import time
import logging
import parse_example
import pandas as pd

import numpy as np
#import matplotlib.pyplot as plt

from idhv import HDModel
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import make_blobs, make_classification

LOG = logging.getLogger(os.path.basename(__file__))
ch = logging.StreamHandler()
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ch.setFormatter(logging.Formatter(log_fmt))
ch.setLevel(logging.INFO)
LOG.addHandler(ch)
LOG.setLevel(logging.INFO)


def main(D, dataset):
    hd_encoding_dim     = D
    #LOG.info("--------- STD: {} ---------".format(cluster_std))
    list = ["Atom","Chainlink","EngyTime","Golfball","Hepta","Lsun","Target","Tetra","TwoDiamonds","WingNut","iris","isolet"]
    sparseList=["100","60","50","40","30","20","10","5","1"]
    dict={"Dataset":{},"100":{},"60":{},"50":{},"40":{},"30":{},"20":{},"10":{},"5":{},"1":{}}

    df=pd.DataFrame(dict)

    for data in list:
        list=[]
        list.append(data)
        for sparsity in sparseList:
            list.append(do_exp(hd_encoding_dim, data,False,int(sparsity)))
        new_row_df = pd.DataFrame([list], columns=df.columns)
        df = pd.concat([df, new_row_df])
        print(df)
    df.to_excel("output.xlsx")
    

def read_data(fn, tag_col = 0, attr_name = False):
    X_ = []
    y_ = []
    with open(fn) as f:
        first_line = True
        for line in f:
            if first_line and attr_name:
                first_line = False
                continue
            data = line.strip().split(',')
            X_.append(data)
            y_.append(data[tag_col])
    return X_, y_

def do_exp(dim, dataset, quantize=False,sparsity=100):
    if(dataset == 'isolet' or dataset == 'iris'):
        train_data_file_name= '../dataset/FCPS/' + dataset + '/' + dataset + '_train.choir_dat'
        nFeatures, nClasses, x_train, y_train = parse_example.readChoirDat(train_data_file_name)
        X_ = x_train
        y_ = y_train
    else:
        X_, y_ = read_data('../dataset/FCPS/%s.csv' % dataset, 0, True)

    X_float = []
    for x in X_:
        X_float.append(list(map(lambda c: float(c), x[1:])))
    X = np.array(X_float)
    y = np.array(list(map(lambda c: float(c) - 1, y_)))
    num_clusters = np.unique(y).shape[0]

    #if num_features == 2:
        #plt.scatter(X[:,0], X[:,1], c=y, s=30, cmap=plt.cm.Paired)

    # K = KMeans(n_clusters = num_clusters, n_init = 5)
    # start = time.time()
    # K.fit(X)
    # end = time.time()
    # kmeans_fit_time = end - start

    # start = time.time()
    # l = K.predict(X)
    # end = time.time()
    # kmeans_predict_time = end - start
    # #LOG.info("K Means Accuracy: {}".format(normalized_mutual_info_score(y, l)))
    # kmeans_score = normalized_mutual_info_score(y, l)
    # kmeans_iter = K.n_iter_

    #M = HDModel(X, y, dim, 100)
    # X = np.asarray(X)
    # L = 100
    # cnt_id = len(X[0])
    # id_hvs = []
    # for i in range(cnt_id):
    #     temp = [-1]*round(D/2) + [1]*round(D/2)
    #     np.random.shuffle(temp)
    #     id_hvs.append(np.asarray(temp))
    # #id_hvs = map(np.int8, id_hvs)
    # lvl_hvs = []
    # temp = [-1]*round(D/2) + [1]*round(D/2)
    # np.random.shuffle(temp)
    # lvl_hvs.append(temp)
    # change_list = list(range(0, D))
    # np.random.shuffle(change_list)
    # cnt_toChange = int(D/2 / (L-1))
    # for i in range(1, L):
    #     temp = np.array(lvl_hvs[i-1])
    #     temp[change_list[(i-1)*cnt_toChange : i*cnt_toChange]] = -temp[change_list[(i-1)*cnt_toChange : i*cnt_toChange]]
    #     lvl_hvs.append(temp)
    # #lvl_hvs = map(np.int8, lvl_hvs)
    # #lvl_hvs = list(map(int,lvl_hvs))
    # x_min = np.min(X)
    # x_max = np.max(X)
    # bin_len = (x_max - x_min)/float(L)
    # start = time.time()
    # train_enc_hvs = encoding(X, lvl_hvs, id_hvs, dim, bin_len, x_min, L)
    # end = time.time()
    # encoding_id_time = end - start
    # #print(encoding_id_time)
    # #M.buildBufferHVs("train", dim)
    # X_h = np.array(train_enc_hvs)

    # KH = KMeans(n_clusters = num_clusters, n_init = 5)
    # start = time.time()
    # KH.fit(X_h)
    # end = time.time()
    # kmeans_hd_fit_time = end - start

    # start = time.time()
    # lh = KH.predict(X_h)
    # end = time.time()
    # kmeans_hd_predict_time = end - start
    # #LOG.info("HD (LV) KMeans Accuracy: {}".format(
    # #    normalized_mutual_info_score(y, lh)))
    # kmeans_hd_score = normalized_mutual_info_score(y, lh)
    # kmeans_hd_iter = KH.n_iter_

    # start = time.time()
    # lh2 = hd_cluster(X_h, num_clusters, quantize=quantize)
    # end = time.time()
    # hd_predict_time = end - start
    # LOG.info("HD (LV) Cluster Accuracy: {}".format(
    #    normalized_mutual_info_score(y, lh2)))
    # hd_score = normalized_mutual_info_score(y, lh2)

    Xb = np.concatenate((X, np.ones((X.shape[0],1))), axis=1)
    PHI = np.random.normal(size=(dim, Xb.shape[1]))

    PHI /= np.linalg.norm(PHI, axis=1).reshape(-1,1)

    start = time.time()
    # X_h = np.sign(PHI.dot(Xb.T).T)
    X_h = (PHI.dot(Xb.T).T)

    end = time.time()
    encoding_phd_time = end - start
    np.sign(make_sparse_standard_deviation(X_h,sparsity))
    KH = KMeans(n_clusters = num_clusters, n_init = 5)
    start = time.time()
    KH.fit(X_h)
    end = time.time()
    kmeans_phd_fit_time = end - start

    # start = time.time()
    # lh = KH.predict(X_h)
    # end = time.time()
    # kmeans_phd_predict_time = end - start
    # LOG.info("HD (RP) KMeans Accuracy: {}".format(
    #    normalized_mutual_info_score(y, lh)))
    # kmeans_phd_score = normalized_mutual_info_score(y, lh)
    # kmeans_phd_iter = KH.n_iter_

    start = time.time()
    lh2 = hd_cluster(X_h, num_clusters, quantize=quantize)
    end = time.time()
    phd_predict_time = end - start
    LOG.info("HD (RP) Cluster Accuracy: {}".format(
       normalized_mutual_info_score(y, lh2)))
    phd_score = normalized_mutual_info_score(y, lh2)

    #print(dim, samples_per_cluster, num_clusters, num_features, cluster_std)
    #synthetic data
    '''
    print(str(dim) + ', ' + str(samples_per_cluster) + ', ' + str(num_clusters) + ', ' + str(num_features) + ', ' + 
        str(cluster_std) + ', ' + str(kmeans_score) + ', ' + str(kmeans_fit_time) + ', ' + str(kmeans_predict_time) + ', ' + 
        str(kmeans_hd_score) + ', ' + str(hd_score) + ', ' + str(encoding_id_time) + ', ' + 
        str(kmeans_hd_fit_time) + ', ' + str(kmeans_hd_predict_time) + ', ' + str(hd_predict_time) + ', ' + 
        str(kmeans_phd_score) + ', ' + str(phd_score) + ', ' + str(encoding_phd_time) + ', ' + 
        str(kmeans_phd_fit_time) + ', ' + str(kmeans_phd_predict_time) + ', ' + str(phd_predict_time))
    '''
    # print(str(dim) + ', ' + dataset + ', ' + str(kmeans_score) + ', ' + str(kmeans_fit_time) + ', ' + str(kmeans_predict_time) + ', ' + 
    #     str(kmeans_hd_score) + ', ' + str(hd_score) + ', ' + str(encoding_id_time) + ', ' + 
    #     str(kmeans_hd_fit_time) + ', ' + str(kmeans_hd_predict_time) + ', ' + str(hd_predict_time) + ', ' + 
    #     str(kmeans_phd_score) + ', ' + str(phd_score) + ', ' + str(encoding_phd_time) + ', ' + 
    #     str(kmeans_phd_fit_time) + ', ' + str(kmeans_phd_predict_time) + ', ' + str(phd_predict_time) + ', ' +
    #     str(kmeans_iter) + ', ' + str(kmeans_hd_iter) + ', ' + str(kmeans_phd_iter))
    return phd_score


def encoding(X_data, lvl_hvs, id_hvs, D, bin_len, x_min, L):
	enc_hv = []
	for i in range(len(X_data)):
		#if i % 100 == 0:
			#print(i)
		sum = np.array([0] * D)
		for j in range(len(X_data[i])):
			bin = min( int((X_data[i][j] - x_min)/bin_len), L-1)
			sum += lvl_hvs[bin]*id_hvs[j]
		enc_hv.append(sum)
	return enc_hv

def hd_cluster(X, num_clusters, max_iter=10, quantize=False):
    scores = []
    for _ in range(max_iter):
        scores.append(hd_cluster_worker(X, num_clusters))
    
    model = sorted(scores, key=lambda x: x[1])[-1]
    return model[0]


def hd_cluster_worker(X, num_clusters, quantize=False):
    C = init_kmpp(X, num_clusters)

    assignments_prev = np.zeros(X.shape[0])
    assignments = compute_similarity(X, C).argmax(axis=1)

    iterations = 0
    while np.sum(assignments != assignments_prev) > 0 and iterations < 100:
        assignments_prev = assignments
        for n in range(num_clusters):
            C[n,:] = X[assignments == n,:].sum(axis=0)
        
        if quantize:
            C = np.sign(C)

        assignments = compute_similarity(X, C).argmax(axis=1)

        # if any cluster has no members randomly sample a point distant
        # from all current cluster centers

        not_missing = np.unique(assignments)
        missing = np.setdiff1d(np.arange(num_clusters), not_missing)
        if missing.size > 0:
            sim = compute_similarity(C[not_missing,:], X).max(axis=0)
            dists = 1/np.clip(sim, 1e-5, np.inf)
            pr = dists / dists.sum()
            for k in missing:
                ix = np.random.choice(X.shape[0], 1, p=pr)
                C[k,:] = X[ix,:]

        iterations += 1
    
    score = 0
    for n in range(num_clusters):
        sub = X[assignments == n,:]
        score += np.mean(compute_similarity(sub, C[n,:].reshape(1,-1)))

    return assignments, score


def init_kmpp(X, num_clusters):
    C = []
    dists = np.ones(X.shape[0])

    cluster_ixs = set([-1])
    for k in range(num_clusters):
        d2 = np.power(dists, 2)
        pr = d2 / np.sum(d2)

        ix = -1
        while ix in cluster_ixs:
            ix = np.random.choice(X.shape[0], 1, p=pr)[0]
        cluster_ixs.update([ix])

        C.append(X[ix,:].reshape(1,-1))
        sim = compute_similarity(np.concatenate(C), X).max(axis=0)
        dists = 1/np.clip(sim, 1e-5, np.inf)

    C = np.concatenate(C)
    return C    


def compute_similarity(X, C):
    X_ = X / np.clip(np.linalg.norm(X, axis=1), 1, np.inf).reshape(-1,1)
    C_ = C / np.clip(np.linalg.norm(C, axis=1), 1, np.inf).reshape(-1,1)
    return np.clip(C_.dot(X_.T).T, 0, np.inf)

def sparse_function(mat,s=5):
        mean_val = np.mean(mat)
        threshold_val = np.percentile(np.abs(mat - mean_val), 5) + mean_val

# Create new sparse array
        sparse_PHI = np.where(np.abs(mat - mean_val) > threshold_val, mat, 0)

# Calculate sparsity
        sparsity = np.count_nonzero(sparse_PHI) / np.size(sparse_PHI)
        print(sparsity)
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                mat[i][j]=mat[i][j] if randint(1,100)<=s else 0
        return mat

def make_sparse_mean(mat,s=5):
    # calculate the mean of each column
    means = np.mean(mat, axis=0)
    # iterate over each column
    for i in range(mat.shape[1]):
        # print(means[i])
        # calculate the absolute difference between each value and the mean
        abs_diff = np.abs(mat[:,i] - means[i])

        # find the s% smallest absolute differences means that we are taking data near to mean
        k = int(s*0.01 * len(abs_diff))
        if k >= len(abs_diff):
            threshold = np.min(abs_diff)
        else:
            # find k largest element
            threshold = np.partition(abs_diff, -k-1)[-k-1] 

        # set all values in the column that are below the threshold to 0
        mat[abs_diff < threshold, i] = 0
    # print(s)
    # print((mat != 0).sum().sum()/mat.size)
    return mat

def make_sparse_standard_deviation(mat,s=5):
    # Step 1: Calculate the standard deviation for each dimension (column)
    standard_deviations = np.std(mat, axis=0)
    k = int(s*0.01 * len(standard_deviations))
    if k >= len(standard_deviations):
            threshold = np.min(standard_deviations)
    else:
            threshold = np.partition(standard_deviations, -k-1)[-k-1]

    non_relevant_dimensions = np.where(standard_deviations < threshold)[0]
    mat[:, non_relevant_dimensions] = 0
    # print(s)
    # print((mat != 0).sum().sum()/mat.size)
    return mat

if __name__=="__main__":
    # if len(sys.argv) != 3:
    #     print('incorrect number of arguments')
    #     print('Usage: ')
    #     print('1st argument: Dataset')
    #     print('2nd argument: Dimensionality')
    #     exit()
    # dataset = sys.argv[1]
    # D = int(sys.argv[2])
    main(10000, "Atom")
