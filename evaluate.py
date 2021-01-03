# file to store the evaluation code of coreset based method performance in machine learning problem
from dataloader import loader, sample
from points import Points
from coreset import compute_coreset


def coreset_evaluate_kmeans(data):
    '''
    Evaluate coreset based method in kmeans clustering problem
    :param data: original data
    :return: performance evaluation result
    '''
    assert type(data).__module__ == 'numpy', 'Please input numpy array as data to conduct the evaluation'
    size, dim = data.shape
    original_points = Points(size, dim)
    original_points.fill_points(data)

    # operate coreset method
    coreset = compute_coreset(data, 5, 1000)
    ori_center, ori_cost = original_points.kmeans_clustering(5)

    cor_center, cor_cost = coreset.kmeans_clustering(5)

    print('Centers obtained from the original set: {}'.format(ori_center))
    print('Centers obtained from the coreset: {}'.format(cor_center))
    print('Sum of squared distances of samples to their closest cluster center in original set: {}'.format(ori_cost))
    print('Sum of squared distances of samples to their closest cluster center in coreset: {}'.format(cor_cost))


def evaluate_1():
    data = loader(filename='Activity recognition exp', specific_file='Watch_gyroscope.csv')
    data = sample(data, size=100000)
    coreset_evaluate_kmeans(data)


if __name__ == '__main__':
    evaluate_1()
