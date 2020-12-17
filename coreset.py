import numpy as np
from cost import compute_kmeans_cost
from points import Points

def distance(p1, p2):
    return np.sum((p1 - p2) ** 2)

def initial_cluster(data, k):
    '''
    initialized the centers for K-means++
    inputs:
        data - numpy array
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


def _compute_sigma_x(points, centers):
    size = points.shape[0]
    dist = np.zeros(size)
    for i in range(size):
        dist[i] = min(distance(c, data[i]) for c in centers)
    total_sum = dist.sum()
    sigma_x = dist / total_sum + 1 / len(centers)
    return sigma_x


def compute_coreset(points, k, N):
    '''
    Implement the core algorithm of generation of coreset
    :param points:weighted points
    :param k: the amount of initialized centers, caculated by k-means++ method
    :param N: size of coresets
    :return: coresets that generated from points
    '''
    data_size, dimension = points.shape
    centers = initial_cluster(points, k)
    sigma_x = _compute_sigma_x(points, centers)
    prob_x = sigma_x / sum(sigma_x)
    samples_idx = np.random.choice(np.arange(data_size), N, p=prob_x)
    samples = np.take(points, samples_idx)
    weights = np.take(prob_x, samples_idx)
    coresets = Points(N, dimension)
    coresets.fill_points(samples, weights)

    return coresets





if __name__ == '__main__':
    data = np.random.randint(0, 100, (10000, 8))
    centers, ids = initial_cluster(data, 5)
    print(centers)
    print(ids)