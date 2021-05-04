import time
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


def cuml_kmeans_csv(csv_data_path, n_clusters=2, csv_weights_path=None, index=True, sample_size=None, header=0):
    # raw_data = loader(filename='gdelt', specific_file='20200513.gkgcounts.csv', sep='\t')
    if index:
        data = cudf.read_csv(csv_data_path, index_col=0, header=header)  #read values
    else:
        data = cudf.read_csv(csv_data_path, header=header)

    if csv_weights_path == None:
        weights = None
    else:
        weights = cudf.read_csv(csv_weights_path, header=header)
    
    data = data.select_dtypes(include=['float64', 'int64']) #filter
    #data = data.sample(100000)  #sample
    if sample_size != None:
        data = data[:sample_size]  #sample
    print("input shape:")
    print(data.shape)

    start_time = time.time()
    print("Calling fit")

    kmeans_float = KMeans(n_clusters=n_clusters)
    kmeans_float.fit(data, sample_weight=weights)
    end_time = time.time()

    cost_time = end_time - start_time
    print("Current data shape:{}, cost time:{}".format(data.shape, cost_time))

    print("labels:")
    print(kmeans_float.labels_)
    print("cluster_centers:")
    print(kmeans_float.cluster_centers_)
    print("sum of squared distances of samples to their closest cluster center")
    print(-kmeans_float.score(data))
    print(222222)
    print(weights)    

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
    #try:
    for n in range(1, int_size + 1):
        amount = items_in_per_G * n
        data = array[:amount + 1]
        kmeans_obj, cost_time = cuml_kmeans(data, n_clusters)
        del kmeans_obj
        print("Current data size: {}G, cost time: {}".format(n, cost_time))
        results.append([n, cost_time])
    #except:
     #   print('Error happened, current data size: {}G'.format(n))
    #else:
     #   print('Experiments conducted successfully!')
    #finally:
     #   print("Finished!")

    if int_size != size:
        results[-1][0] = round(size, 1)
    print(results)
    return results


if __name__ == '__main__':
    #results = cuml_speed_experiment('./data/all-latest.npy', 28, 5)
    #np.save('./data/result', np.array(results))
    #write_list(results, './data/results.txt')
    cuml_kmeans_csv('./data/denmark-latest.csv', 5, index=False)
    #cuml_kmeans_csv('/scratch/work/wux4/thesis/ModernCoreset/data/Activity recognition exp/Watch_gyroscope.csv', 5)
    cuml_kmeans_csv('./output/coreset_v.csv', 5, csv_weights_path='./output/coreset_w.csv', index=False, header=None)
