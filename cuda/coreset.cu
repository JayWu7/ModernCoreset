//
// Created by Xiaobo Wu on 2021/2/1.
//
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>

typedef unsigned long long int size_int;
#define FLOAT_MAX_VALUE 3.40282e+038
using namespace std;

__device__ float _squared_l2_distance(thrust::device_vector<float> &first, thrust::device_vector<float> &second) {
    float distance = 0.0;
    int dimension = first.size();
    if (dimension != second.size()) {
        throw "Input two vector are not in the same dimension!";
    }
    for (int i = 0; i < dimension; i++) {
        float gap = first[i] - second[i]
        distance += gap * gap;
    }
    return distance;
}


//todo, test if using [long double] type of dist_sum will influence the performance
__global__ void _min_dist(float dist_sum, thrust::device_vector<float> &last_center,
                          thrust::device_vector <thrust::device_vector<float>> &points,
                          thrust::device_vector<float> &dist, size_int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        float new_dis = _squared_l2_distance(last_center, points[tid]);
        if (new_dis < dist[tid]) {
            dist[tid] = new_dis;
        }

        dist_sum += dist[tid];
    }
}


__global__ void
_calculate_weights(float &dist_sum, thrust::device_vector<float> &weights, thrust::device_vector<float> &dist,
                   size_int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n)
        weights[tid] = dist[tid] / dist_sum;
}


thrust::device_vector <thrust::device_vector<float>>
KMeans::KMeans_pp_Init(vector <vector<float>> &points, int n_cluster) {
    if (n_cluster < 1) {
        throw "n_cluster, the number of clusters should at least greater than 1.";
    }

    thrust::device_vector <thrust::device_vector<float>> centers(n_cluster);
    thrust::device_vector <thrust::device_vector<float>> device_points(points);
    size_int size = points.size();
    thrust::device_vector<float> dist(size, FLOAT_MAX_VALUE);
    thrust::device_vector<float> weights(size);
    float dist_sum; // todo, try [long double] type

    //Choose one center uniformly at random among the data points.
    random_device rd;  //Will be used to obtain a seed for the random number engine
    mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    uniform_int_distribution<> distrib(0, size - 1);
    int first_center_index = distrib(gen);
    centers[0] = device_points[first_center_index];

    int cur_center_index;
    for (int i = 0; i < n_cluster - 1; i++) {
        dist_sum = 0.0;
        // launching kernel
        _min_dist<<<100, 256>>>(dist_sum, centers[i], device_points, dist, size);
        _calculate_weights<<<100, 256>>>(dist_sum, weights, dist, size);

        discrete_distribution<> dd(weights.begin(), weights.end());
        cur_center_index = dd(gen);
        centers[i + 1] = device_points[cur_center_index];
    }
    return centers;
}