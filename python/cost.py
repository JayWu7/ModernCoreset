import numpy as np
from sklearn.cluster import KMeans


def compute_cost(center, points):
    return sum(np.linalg.norm(center - point) ** 2 for point in points)


def kmeans_cost(points, k, seed=16):
    kmeans = KMeans(n_clusters=k, random_state=seed).fit(points)
    return kmeans.cluster_centers_, kmeans.inertia_
