import sys
from cuda_kmeans import cuml_kmeans_csv

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Parameters error!')
        print("Usage: 'python cuml_evaluate.py <original_data_path> <coreset_path> <coreset_weights_path> <cluster_size>'")
        exit() 
    data_path = sys.argv[1]
    coreset_path = sys.argv[2]
    coreset_weights_path = sys.argv[3]
    cluster_size = int(sys.argv[4])
    
    sample_size = None
    if len(sys.argv) > 5: #optional parameters
        sample_size = int(sys.argv[5])
        

    cuml_kmeans_csv(data_path, cluster_size, sample_size=sample_size)
    cuml_kmeans_csv(coreset_path, cluster_size, csv_weights_path=coreset_weights_path)
    
    





