import os
from random import randint
import time
import logging
import parse_example
import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from emnist import extract_training_samples, extract_test_samples

from sklearn.datasets import fetch_openml
from emnist import extract_training_samples, extract_test_samples

import matplotlib.pyplot as plt

from idhv import HDModel
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import make_blobs, make_classification
from tensorflow import keras

LOG = logging.getLogger(os.path.basename(__file__))
ch = logging.StreamHandler()
fh = logging.FileHandler('my_log_file.log')  # Specify the file name here
LOG.addHandler(fh)
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ch.setFormatter(logging.Formatter(log_fmt))
ch.setLevel(logging.INFO)
LOG.addHandler(ch)
LOG.setLevel(logging.INFO)


def main(D, dataset):
    hd_encoding_dim = D
    #LOG.info("--------- STD: {} ---------".format(cluster_std))
    # list = ["Atom","Chainlink","EngyTime","Golfball","Hepta","Lsun","Target","Tetra","TwoDiamonds","WingNut","iris","isolet"]
    # sparseList=["100","90","80","70","60","50","40","30","20","10","5","1"]
    # dict={"Dataset":{},"100":{},"90":{},"80":{},"70":{},"60":{},"50":{},"40":{},"30":{},"20":{},"10":{},"5":{},"1":{}}
    list = ["emnist_resnet_cc_128"]
    sparseList = ["100"]
    dict = {"Dataset": {}, "100": {}}

    df = pd.DataFrame(dict)

    # model,  y_combined ,x_combined= get_mnist_model_from_cnn()
    # layer_no=-1
    # out=""
    # for layer in model.layers:
    #     layer_no+=1
    #     feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)
    #     feat=feature_extractor.predict(x_combined).reshape(70000,-1)
    #     z=do_exp(hd_encoding_dim, "data",False,100,feat,y_combined)
    #     out+="layer "+layer+" final :"+z
    #     out+="feature_extractor "+len(feat)
    #     with open('result.txt', 'w') as file:
    #       file.write(out)

    for data in list:
        lists = []
        lists.append(data)
        for sparsity in sparseList:
            lists.append(do_exp(hd_encoding_dim, data, False,
                                int(sparsity), [], [], True))
        new_row_df = pd.DataFrame([lists], columns=df.columns)
        df = pd.concat([df, new_row_df])
        df.to_excel("baseline_cc_mnist.xlsx")
        print(df)
    for data in list:
        lists = []
        lists.append(data)
        for sparsity in sparseList:
            lists.append(do_exp(hd_encoding_dim, data, False,
                                int(sparsity), [], [], False))
        new_row_df = pd.DataFrame([lists], columns=df.columns)
        df = pd.concat([df, new_row_df])
        df.to_excel("hdrp_cc_mnist.xlsx")
        print(df)


def read_data(fn, tag_col=0, attr_name=False):
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


def genearate_mnist_csv():

    # Load the MNIST dataset from scikit-learn
    mnist = fetch_openml('mnist_784')

    # Separate features (images) and labels
    X = mnist.data
    y = mnist.target.astype(int)

    # Combine the flattened images and labels
    data = np.column_stack((y, X))
    df = pd.DataFrame(data)
    df.to_csv('Mnist.csv', index=False)


def genearate_fasion_mnist_csv():

    # Load the MNIST dataset from scikit-learn
    mnist = fetch_openml('Fashion-MNIST', version=1)

    # Separate features (images) and labels
    X = mnist.data
    y = mnist.target.astype(int)

    # Combine the flattened images and labels
    data = np.column_stack((y, X))
    df = pd.DataFrame(data)
    df.to_csv('FashionMnist.csv', index=False)


def rgb_to_grayscale(rgb):
    return 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]


def convert_to_grayscale(X_):
    matrix_grayscale = []
    for row in X_:
        grayscale_row = []
        for feature in row:
            for pixel in feature:
                grayscale_value = rgb_to_grayscale(pixel)
                grayscale_row.append(grayscale_value)
        matrix_grayscale.append(grayscale_row)
    return matrix_grayscale


def generate_cnn_128_emnist_csv():
    x_train, y_train = extract_training_samples('letters')

    # Load EMNIST letters testing data
    x_test, y_test = extract_test_samples('letters')

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encode the labels
    num_classes = 26
    print(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train-1, num_classes)
    y_test = keras.utils.to_categorical(y_test-1, num_classes)

    # Build the CNN model
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # Print model summary
    model.summary()

    # Train the model
    batch_size = 128
    epochs = 10
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(x_test, y_test))

    # Evaluate the model on the test data
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    feature_extractor = keras.Model(
        inputs=model.inputs, outputs=model.layers[-2].output)

    # Extract features from the training set
    train_features = feature_extractor.predict(x_train)
    print("Train Features shape:", train_features.shape)
    print("Test Features shape:", test_features.shape)
    print("y_train  shape:", y_train.shape)
    print("y_test  shape:", y_test.shape)

    # Assuming you have the following variables:
    # train_features, test_features, y_train, y_test

    # Combine y_train and y_test vertically
    y_combined = np.concatenate((y_train, y_test))

    # Combine train_features and test_features horizontally
    features_combined = np.concatenate((train_features, test_features), axis=0)
    print(features_combined.shape)
    # Create a list of column names
    column_names = ['y'] + \
        [f'feature_{i+1}' for i in range(features_combined.shape[1])]
    print(len(column_names))
    print(y_combined.shape)
    # Create a DataFrame from the combined data
    df = pd.DataFrame(np.column_stack(
        (np.argmax(y_combined, axis=1), features_combined)), columns=column_names)

    # Save the DataFrame to a CSV file
    df.to_csv('Emnist_8_layers.csv', index=False)
    return


def do_exp(dim, dataset, quantize=False, sparsity=100, X_=[], y_=[], k=False):
    if(dataset == 'isolet' or dataset == 'iris'):
        train_data_file_name = '../dataset/FCPS/' + \
            dataset + '/' + dataset + '_train.choir_dat'
        nFeatures, nClasses, x_train, y_train = parse_example.readChoirDat(
            train_data_file_name)
        X_ = x_train
        y_ = y_train
    elif (dataset == 'Mnist'):
        genearate_mnist_csv()
        X_, y_ = read_data('%s.csv' % dataset, 0, True)
    elif (dataset == 'FashionMnist'):
        genearate_fasion_mnist_csv()
        X_, y_ = read_data('%s.csv' % dataset, 0, True)
    elif (dataset == 'cfar_10'):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        # Concatenate training and testing data
        X_ = np.concatenate((x_train, x_test), axis=0)
        y_ = np.concatenate((y_train, y_test), axis=0)
        X_ = X_.tolist()
        y_ = y_.tolist()
        X_ = convert_to_grayscale(X_)
        array = np.array(y_)
        reshaped_array = array.reshape(60000)
        y_ = reshaped_array.tolist()
    elif (dataset == 'Emnist_8_layers'):
        generate_cnn_128_emnist_csv()
        X_, y_ = read_data('%s.csv' % dataset, 0, True)
    else:
        X_, y_ = read_data('../dataset/FCPS/%s.csv' % dataset, 0, True)

    X_float = []

    for x in X_:
        X_float.append(list(map(lambda c: float(c), x[1:])))
    X = np.array(X_float)
    y = np.array(list(map(lambda c: float(c) - 1, y_)))
    num_clusters = np.unique(y).shape[0]
    print(X.shape)
    print(y.shape)

    # if num_features == 2:
    #plt.scatter(X[:,0], X[:,1], c=y, s=30, cmap=plt.cm.Paired)
    if(k):
        K = KMeans(n_clusters=num_clusters, n_init=5)
        start = time.time()
        K.fit(X)
        end = time.time()
        kmeans_fit_time = end - start

        start = time.time()
        l = K.predict(X)
        end = time.time()
        kmeans_predict_time = end - start
        kmeans_score = normalized_mutual_info_score(y, l)
        LOG.info("K Means Accuracy " + dataset + ": {}".format(kmeans_score))
        return kmeans_score

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
    else:
        Xb = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        PHI = np.random.normal(size=(dim, Xb.shape[1]))
        PHI /= np.linalg.norm(PHI, axis=1).reshape(-1, 1)
        # random_sparse_function(PHI,sparsity)
        start = time.time()
        X_h = np.sign(PHI.dot(Xb.T).T)
        # X_h = (PHI.dot(Xb.T).T)
        end = time.time()
        encoding_phd_time = end - start
        # np.sign(make_sparse_standard_deviation(X_h,sparsity))
        # KH = KMeans(n_clusters = num_clusters, n_init = 5)
        # start = time.time()
        # KH.fit(X_h)
        # end = time.time()
        # kmeans_phd_fit_time = end - start

        # start = time.time()
        # lh = KH.predict(X_h)
        # end = time.time()
        # kmeans_phd_predict_time = end - start
        # LOG.info("HD (RP) KMeans Accuracy: {}".format(
        #    normalized_mutual_info_score(y, lh)))
        # kmeans_phd_score = normalized_mutual_info_score(y, lh)
        # kmeans_phd_iter = KH.n_iter_
        # return kmeans_phd_score
        # start = time.time()
        lh2 = hd_cluster(X_h, num_clusters, quantize=quantize)
        end = time.time()
        phd_predict_time = end - start
        LOG.info("HD (RP) Cluster Accuracy " + dataset + ": {}".format(
            normalized_mutual_info_score(y, lh2)))
        phd_score = normalized_mutual_info_score(y, lh2)

        #print(dim, samples_per_cluster, num_clusters, num_features, cluster_std)
        # synthetic data
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
        # if i % 100 == 0:
        # print(i)
        sum = np.array([0] * D)
        for j in range(len(X_data[i])):
            bin = min(int((X_data[i][j] - x_min)/bin_len), L-1)
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
    print("hd")
    C = init_kmpp(X, num_clusters)

    assignments_prev = np.zeros(X.shape[0])
    assignments = compute_similarity(X, C).argmax(axis=1)

    iterations = 0
    while np.sum(assignments != assignments_prev) > 0 and iterations < 100:
        assignments_prev = assignments
        for n in range(num_clusters):
            C[n, :] = X[assignments == n, :].sum(axis=0)

        if quantize:
            C = np.sign(C)

        assignments = compute_similarity(X, C).argmax(axis=1)

        # if any cluster has no members randomly sample a point distant
        # from all current cluster centers

        not_missing = np.unique(assignments)
        missing = np.setdiff1d(np.arange(num_clusters), not_missing)
        if missing.size > 0:
            sim = compute_similarity(C[not_missing, :], X).max(axis=0)
            dists = 1/np.clip(sim, 1e-5, np.inf)
            pr = dists / dists.sum()
            for k in missing:
                ix = np.random.choice(X.shape[0], 1, p=pr)
                C[k, :] = X[ix, :]

        iterations += 1

    score = 0
    for n in range(num_clusters):
        sub = X[assignments == n, :]
        score += np.mean(compute_similarity(sub, C[n, :].reshape(1, -1)))

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

        C.append(X[ix, :].reshape(1, -1))
        sim = compute_similarity(np.concatenate(C), X).max(axis=0)
        dists = 1/np.clip(sim, 1e-5, np.inf)

    C = np.concatenate(C)
    return C


def compute_similarity(X, C):
    X_ = X / np.clip(np.linalg.norm(X, axis=1), 1, np.inf).reshape(-1, 1)
    C_ = C / np.clip(np.linalg.norm(C, axis=1), 1, np.inf).reshape(-1, 1)
    return np.clip(C_.dot(X_.T).T, 0, np.inf)


def random_sparse_function(mat, s=5):
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            mat[i][j] = mat[i][j] if randint(1, 100) <= s else 0
    # print(s)
    # print((mat != 0).sum().sum()/mat.size)
    return mat


def make_sparse_mean(mat, s=5):
    # calculate the mean of each column
    means = np.mean(mat, axis=0)
    # iterate over each column
    for i in range(mat.shape[1]):
        # print(means[i])
        # calculate the absolute difference between each value and the mean
        abs_diff = np.abs(mat[:, i] - means[i])

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


def make_sparse_standard_deviation(mat, s=5):
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

# def get_mnist_model_from_cnn():

#     # Load and preprocess the MNIST dataset
#     (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#     x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
#     x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
#     num_classes = 10

#     model = keras.Sequential(
#         [
#             layers.Conv2D(5, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
#             layers.MaxPooling2D(pool_size=(2, 2)),
#             layers.Conv2D(5, kernel_size=(3, 3), activation="relu"),
#             layers.MaxPooling2D(pool_size=(2, 2)),
#             layers.Flatten(),
#             layers.Dense(128, activation="relu"),  # Additional dense layer
#             layers.Dropout(0.5),
#             layers.Dense(num_classes, activation="softmax"),
#         ]
#     )
#     # Compile and train the model
#     model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#     model.fit(x_train, y_train, batch_size=128, epochs=20, validation_split=0.1)
#     return model,np.concatenate((y_train, y_test)),np.concatenate((x_train, x_test))


if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print('incorrect number of arguments')
    #     print('Usage: ')
    #     print('1st argument: Dataset')
    #     print('2nd argument: Dimensionality')
    #     exit()
    # dataset = sys.argv[1]
    # D = int(sys.argv[2])
    main(10000, "Atom")
