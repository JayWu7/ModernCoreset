import numpy as np


def compute_cost(center, points):
    return sum(np.linalg.norm(center - point) for point in points)
