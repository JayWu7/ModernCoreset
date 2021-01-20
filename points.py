import numpy as np
from sklearn.cluster import KMeans


# class Point:
#     def __init__(self, value, weight=1):
#         self.value = value
#         self.dimension = len(value)
#         self.weight = weight


class Points:
    def __init__(self, size, dimension, seed=16):
        self.size = size
        self.dimension = dimension
        self.values = np.zeros((size, dimension))
        self.weights = np.ones(size)  # initialize all points weight equal to 1
        self.seed = seed

    def __len__(self):
        return self.size

    def set_seed(self, new_seed):
        self.seed = new_seed

    def fill_points(self, values, weights=None):
        assert values.shape == self.values.shape, 'Please input values with the shape of {}'.format(self.values.shape)
        self.values = values
        if weights is not None:
            assert weights.shape == self.weights.shape, 'Please input weights with the shape of {}'.format(
                self.weights.shape)
            self.set_weights(weights)

    def add_points(self, values, weights=None):
        assert self.values.shape[1] == values.shape[
            1], 'Please add the points with same dimension {} as current points'.format(self.values.shape[1])

        self.values = np.concatenate([self.values, values], axis=0)
        if weights is not None:
            assert len(weights) == len(values), 'The new values and weights are not in the same length.'
            self.weights = np.concatenate([self.weights, weights], axis=0)
        else:
            self.weights = np.concatenate([self.weights, np.ones(len(values))], axis=0)

        self.size += len(values)

    def set_weights(self, weights):
        assert len(weights) == len(self.values), 'The new weights are not in the same length with the values.'
        self.weights = weights

    def get_values(self):
        return self.values

    def get_dimension(self):
        return self.dimension

    def get_weights(self):
        return self.weights

    def kmeans_clustering(self, k):
        kmeans = KMeans(n_clusters=k, random_state=self.seed).fit(self.values, sample_weight=self.weights)
        return kmeans.cluster_centers_, kmeans.inertia_
