# Both import methods supported
from cuml import KMeans
from cuml.cluster import KMeans
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


def cuml_kmeans():
    raw_data = loader(filename='gdelt', specific_file='20200513.gkgcounts.csv', sep='\t')
    data = np2cudf(raw_data)
    print("input:")
    print(data)

    print("Calling fit")
    kmeans_float = KMeans(n_clusters=2)
    kmeans_float.fit(data)

    print("labels:")
    print(kmeans_float.labels_)
    print("cluster_centers:")
    print(kmeans_float.cluster_centers_)
