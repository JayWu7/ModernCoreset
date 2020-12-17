import numpy as np


def compute_kmeans_cost(center, points):
    return sum(np.linalg.norm(center - point) ** 2 for point in points)




