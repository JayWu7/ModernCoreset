import numpy as np

class Point:
    def __init__(self, value, weight=1):
        self.value = value
        self.dimension = len(value)
        self.weight = weight


class Points:
    def __init__(self, size, dimension):
        self.size = size
        self.dimension = dimension
        self.values = np.zeros((size, dimension))
        self.weights = np.ones(size)

    def __len__(self):
        return self.size








