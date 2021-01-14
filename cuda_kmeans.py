# Both import methods supported
from cuml import KMeans
from cuml.cluster import KMeans

import cudf
import numpy as np
import pandas as pd

def np2cudf(df):
    # convert numpy array to cuDF dataframe
    df = pd.DataFrame({'fea%d'%i:df[:,i] for i in range(df.shape[1])})
    pdf = cudf.DataFrame()
    for c,column in enumerate(df):
      pdf[str(c)] = df[column]
    return pdf

a = np.asarray([[1.0, 1.0], [1.0, 2.0], [3.0, 2.0], [4.0, 3.0]],
               dtype=np.float32)
b = np2cudf(a)
print("input:")
print(b)

print("Calling fit")
kmeans_float = KMeans(n_clusters=2)
kmeans_float.fit(b)

print("labels:")
print(kmeans_float.labels_)
print("cluster_centers:")
print(kmeans_float.cluster_centers_)
