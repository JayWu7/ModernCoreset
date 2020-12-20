import numpy as np
from points import Points
from sklearn.cluster import KMeans


class Point:
    def __init__(self, value, weight=1):
        self.value = value
        self.dimension = len(value)
        self.weight = weight


class Points:
    def __init__(self, size, dimension, seed=16):
        self.size = size
        self.dimension = dimension
        self.values = np.zeros((size, dimension))
        self.weights = np.ones(size)
        self.seed = seed

    def __len__(self):
        return self.size

    def set_seed(self, new_seed):
        self.seed = new_seed

    def fill_points(self, values, weights):
        self.values = values
        self.weights = weights

    def get_values(self):
        return self.values

    def get_weights(self):
        return self.weights

    def kmeans_clustering(self, k):
        kmeans = KMeans(n_clusters=k, random_state=self.seed).fit(self.values, sample_weight=self.weights)
        return kmeans.cluster_centers_
