import time
import os
from cuml import KMeans
import cudf
import numpy as np
import pandas as pd
from dataloader import loader, sample
from tools import write_list
import dask_cudf


def np2cudf(df):
    # convert numpy array to cuDF dataframe
    df = pd.DataFrame({'fea%d' % i: df[:, i] for i in range(df.shape[1])})
    pdf = cudf.DataFrame()
    for c, column in enumerate(df):
        pdf[str(c)] = df[column]
    pdf = dask_cudf.from_cudf(pdf,npartitions=6)
    return pdf


def cuml_kmeans(raw_data, n_clusters=2):
    # raw_data = loader(filename='gdelt', specific_file='20200513.gkgcounts.csv', sep='\t')
    data = np2cudf(raw_data)
    print("input shape:")
    print(data.shape)

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


def cuml_kmeans_csv(csv_data_path, n_clusters=2, csv_weights_path=None, index=False, sample_size=None, header=0, store_path="./output"):
    filename = csv_data_path.split('/')[-1][:-4]  #get the file name
    if index:
        data = cudf.read_csv(csv_data_path, index_col=0, header=header, dtype=float)  #read values
    else:
        data = cudf.read_csv(csv_data_path, header=header, dtype=float)

    if csv_weights_path == None:
        weights = None
    else:
        weights = cudf.read_csv(csv_weights_path, header=header)
    
    #data = data.sample(100000)  #sample
    if sample_size != None:
        data = data[:sample_size]  #sample

    start_time = time.time()
    print("Calling kmeans fit function!")
    
    kmeans_float = KMeans(n_clusters=n_clusters)
    kmeans_float.fit(data, sample_weight=weights)
    end_time = time.time()

    cost_time = end_time - start_time
    print("Current data shape:{}, cost time for kmeans clustering:{}".format(data.shape, cost_time))

    labels = kmeans_float.labels_.to_frame()  # Get labels
    labels_path = os.path.join(store_path, filename + '-labels' + '.csv')
    labels.to_csv(labels_path, index=False)
    print('Clustering result labels stored in: {}'.format(labels_path))

    centers = kmeans_float.cluster_centers_
    centers_path = os.path.join(store_path, filename + '-centers' + '.csv')
    centers.to_csv(centers_path, index=False)
    print('Clustering centers stored in: {}'.format(centers_path))


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
    #try:
    for n in range(1, int_size + 1):
        amount = items_in_per_G * n
        data = array[:amount + 1]
        kmeans_obj, cost_time = cuml_kmeans(data, n_clusters)
        del kmeans_obj
        print("Current data size: {}G, cost time: {}".format(n, cost_time))
        results.append([n, cost_time])

    if int_size != size:
        results[-1][0] = round(size, 1)
    print(results)
    return results


def evaluate_1(data_path, coreset_path, coreset_weights_path):
    cuml_kmeans_csv('./data/denmark-latest.csv', 5, index=False)
    cuml_kmeans_csv('./output/coreset_v.csv', 5, csv_weights_path='./output/coreset_w.csv', index=False, header=None)
    




if __name__ == '__main__':
    #results = cuml_speed_experiment('./data/all-latest.npy', 28, 5)
    #np.save('./data/result', np.array(results))
    #write_list(results, './data/results.txt')
    cuml_kmeans_csv('./data/Watch_gyroscope.csv', 50, index=False, has_non_numeric=True)
    #cuml_kmeans_csv('/scratch/work/wux4/thesis/ModernCoreset/data/Activity recognition exp/Watch_gyroscope.csv', 5)
    cuml_kmeans_csv('./output/coreset_v.csv', 50, csv_weights_path='./output/coreset_w.csv', index=False)
