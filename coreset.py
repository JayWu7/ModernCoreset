import numpy as np


def distance(p1, p2):
    return np.sum((p1 - p2) ** 2)


def initial_cluster(data, k):
    '''
    initialized the centers for K-means++
    inputs:
        data - numpy array of data points having shape (200, 2)
        k - number of clusters
    '''
    centers = []
    centers_indices = []
    size = data.shape[0]
    dist = np.zeros(size)
    indices = np.arange(size)
    # plot(data, np.array(centers))

    first_center_id = np.random.choice(indices, 1)[0]
    first_center = data[first_center_id]
    centers_indices.append(first_center_id)
    centers.append(first_center)

    for _ in range(k - 1):
        for i in range(size):
            dist[i] = min(distance(c, data[i]) for c in centers)  # Improvement can be done here
        weights = dist / sum(dist)
        ## select data point with maximum distance as our next centroid
        next_center_id = np.random.choice(indices, 1, p=weights)[0]
        next_center = data[next_center_id]
        centers_indices.append(next_center_id)
        centers.append(next_center)
        # plot(data, np.array(centers))

    return centers, centers_indices


def compute_coreset(points):
    # todo
    pass


if __name__ == '__main__':
    data = np.random.randint(0, 100, (10000, 8))
    centers, ids = initial_cluster(data, 5)
    print(centers)
    print(ids)