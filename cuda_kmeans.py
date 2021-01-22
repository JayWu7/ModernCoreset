import time
from cuml import KMeans
from dataloader import loader

import cudf
import numpy as np
import pandas as pd


def np2cudf(df):
    # convert numpy array to cuDF dataframe
    df = pd.DataFrame({'fea%d' % i: df[:, i] for i in range(df.shape[1])})
    pdf = cudf.DataFrame()
    for c, column in enumerate(df):
        pdf[str(c)] = df[column]
    return pdf


def cuml_kmeans(raw_data, n_clusters=2):
    # raw_data = loader(filename='gdelt', specific_file='20200513.gkgcounts.csv', sep='\t')
    data = np2cudf(raw_data)
    print("input:")
    print(data)

    start_time = time.time()
    print("Calling fit")

    kmeans_float = KMeans(n_clusters=n_clusters)
    kmeans_float.fit(data)
    end_time = time.time()

    cost_time = end_time - start_time
    print("Current data shape:{}, cost time:{}".format(raw_data.shape, cost_time))

    print("labels:")
    print(kmeans_float.labels_)
    print("cluster_centers:")
    print(kmeans_float.cluster_centers_)

    return kmeans_float, cost_time


def cuml_speed_experiment(np_file, size, n_clusters=2):
    '''
    This is a function to test the speed performance of cuml in different input size.
    :param np_file: input numpy npy file
    :param size: the size of data in gigabyte form
    :return: list of the running time along with the different data size
    '''
    array = np.load(np_file)
    array_length = array.shape[0]
    items_in_per_G = round(array_length / size)
    results = []
    int_size = int(size) if int(size) == size else int(size) + 1
    for n in range(1, int_size + 1):
        amount = items_in_per_G * n
        data = array[:amount + 1]
        kmeans_obj, cost_time = cuml_kmeans(data, n_clusters)
        results.append([n, cost_time])

    if int_size != size:
        results[-1][0] = round(size, 1)

    return results

