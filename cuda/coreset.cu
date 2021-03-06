//
// Created by Xiaobo Wu on 2021/2/1.
//
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>
#include "points.h"


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


__device__ float _squared_l2_distance_with_index(thrust::device_vector<float> &first, size_int fid, thrust::device_vector<float> &second, size_int sid, unsigned int dimension) {
    float distance = 0.0;
    for (int i = 0; i < dimension; i++) {
        float gap = first[fid + i] - second[sid + i]
        distance += gap * gap;
    }
    return distance;
}


//todo, test if using [long double] type of dist_sum will influence the performance
__global__ void _min_dist(float dist_sum, thrust::device_vector<float> &centers, size_int center_id, 
                          thrust::device_vector<float> &device_points,
                          thrust::device_vector<float> &dist, size_int n, unsigned int dimension) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        int sid = tid * dimension;
        float new_dis = _squared_l2_distance_with_index(centers, center_id, device_points, sid, dimension);
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


//Change the function to accept 1-d form input
thrust::device_vector<float>
k_means_pp_init_cu(thrust::device_vector<float> &device_points, int n_cluster, unsigned int dimension) {
    if (n_cluster < 1) {
        throw "n_cluster, the number of clusters should at least greater than 1.";
    }

    thrust::device_vector<float> centers(n_cluster * dimension); // 1 d
    size_int size = points.size() / dimension;
    thrust::device_vector<float> dist(size, FLOAT_MAX_VALUE);
    thrust::device_vector<float> weights(size);
    float dist_sum; // todo, try [long double] type

    //Choose one center uniformly at random among the data points.
    random_device rd;  //Will be used to obtain a seed for the random number engine
    mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    uniform_int_distribution<> distrib(0, size - 1);
    int first_center_index = distrib(gen);
    int start_index = first_center_index * dimension;
    
    for (int i = 0; i < dimension; i++){
        centers[i] = device_points[start_index + i];
    }

    int cur_center_index;
    for (int i = 0; i < n_cluster - 1; i++) {
        dist_sum = 0.0;
        // launching kernel
        int center_start_id = i * dimension;
        _min_dist<<<100, 256>>>(dist_sum, centers, center_start_id; device_points, dist, size, dimension);
        _calculate_weights<<<100, 256>>>(dist_sum, weights, dist, size);

        discrete_distribution<> dd(weights.begin(), weights.end());
        cur_center_index = dd(gen);
        start_index = cur_center_index * dimension;
        center_start_id += dimension; 
        for (int i = 0; i < dimension; i++){
            centers[center_start_id + i] = device_points[start_index + i];
        }
    }
    return centers;
}


// thrust::device_vector <thrust::device_vector<float>>
// k_means_pp_init_cu(vector <vector<float>> &points, int n_cluster) {
//     if (n_cluster < 1) {
//         throw "n_cluster, the number of clusters should at least greater than 1.";
//     }

//     thrust::device_vector <thrust::device_vector<float>> centers(n_cluster);
//     thrust::device_vector <thrust::device_vector<float>> device_points(points);
//     size_int size = points.size();
//     thrust::device_vector<float> dist(size, FLOAT_MAX_VALUE);
//     thrust::device_vector<float> weights(size);
//     float dist_sum; // todo, try [long double] type

//     //Choose one center uniformly at random among the data points.
//     random_device rd;  //Will be used to obtain a seed for the random number engine
//     mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
//     uniform_int_distribution<> distrib(0, size - 1);
//     int first_center_index = distrib(gen);
//     centers[0] = device_points[first_center_index];

//     int cur_center_index;
//     for (int i = 0; i < n_cluster - 1; i++) {
//         dist_sum = 0.0;
//         // launching kernel
//         _min_dist<<<100, 256>>>(dist_sum, centers[i], device_points, dist, size);
//         _calculate_weights<<<100, 256>>>(dist_sum, weights, dist, size);

//         discrete_distribution<> dd(weights.begin(), weights.end());
//         cur_center_index = dd(gen);
//         centers[i + 1] = device_points[cur_center_index];
//     }
//     return centers;
// }


//todo, test if using [long double] type of dist_sum will influence the performance
__global__ void _compute_sigma_dist(thrust::device_vector<float> &device_points,
                                    thrust::device_vector<float> &centers,
                                    thrust::device_vector<float> &dist, thrust::device_vector<int> &assign,
                                    thrust::device_vector<int> &cluster_size, float &dist_sum, size_int n, 
                                    unsigned int n_cluster, unsigned int dimension) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int center_id;
    float min_dist = FLOAT_MAX_VALUE;
    if (tid < n) {
        for (int i = 0; i < n_cluster; i++) {
            int center_start_id = i * dimension; 
            int sid = tid * dimension;
            float cur_dist = _squared_l2_distance_with_index(centers, center_start_id, device_points, sid, dimension);
            if (cur_dist < min_dist) {
                min_dist = cur_dist;
                center_id = i;
            }
        }
        dist[tid] = min_dist;
        dist_sum += min_dist;
        assign[tid] = center_id;
        cluster_size[center_id] += 1;
    }
}


__global__ void _compute_sigma_x(thrust::device_vector<float> &dist,
                                 thrust::device_vector<int> &assign,
                                 thrust::device_vector<int> &cluster_size, float &dist_sum, float &sigma_sum, size_int n) {


    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float res = dist[tid] / dist_sum + 1 / cluster_size[assign[tid]];
        dist[tid] = res; // change dist vector to sigma_x vector
        sigma_sum += res;
    }
}

__global__ void _compute_prob_x(thrust::device_vector<float> &sigma_x, float sigma_sum, size_int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        sigma_x[tid] = sigma_x[tid] / sigma_sum; // change sigma_x vector to prob_x vector
    }
}


__global__ void _compute_weights(thrust::device_vector<float> &prob_x, size_int n, size_int n_coreset) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        prob_x[tid] = 1 / (n_coreset * prob_x[tid]); // change prob_x vector to weight_x vector
    }
}


thrust::device_vector<float>
_compute_sigma(thrust::device_vector<float> &device_points,
               thrust::device_vector<float> &centers, size_int n, 
               unsigned int n_cluster, unsigned int dimension) {

    thrust::device_vector<float> dist(n, 0.0);
    thrust::device_vector<unsigned int> assign(n);
    thrust::device_vector<size_int> cluster_size(n_cluster);

    float dist_sum = 0.0; // todo, test using long double
    float sigma_sum = 0.0;

    _compute_sigma_dist<<<100, 256>>>(device_points, centers, dist, assign, cluster_size, dist_sum, n, n_cluster, dimension);
    _compute_sigma_x<<<100, 256>>>(dist, assign, cluster_size, dist_sum, sigma_sum, n);
    _compute_prob_x<<<100, 256>>>(dist, sigma_sum, n);  
    
    // dist here is the prob_x
    return dist;
}



// Compute Coreset Function, return the Points type 
coreset::Points
compute_coreset(vector<float> &points, unsigned int dimension, unsigned int n_cluster, size_int n_coreset) {
    size_int data_size = points.size() / dimension;
    if (data_size < n_coreset) {
        throw "Setting size of coreset is greater or equal to the original data size, please alter it";
    }

    thrust::device_vector<float> device_points(points.begin(), points.end());

    thrust::device_vector<float> centers; // 1d data
    centers = k_means_pp_init_cu(device_points, n_cluster, unsigned int dimension);

    thrust::device_vector<float> prob_x;
    prob_x = _compute_sigma(device_points, centers, data_size, n_cluster, unsigned int dimension);

    _compute_weights(prob_x, data_size, n_coreset);

    // random_device rd;
    // mt19937 gen(rd());
    // discrete_distribution<> d(prob_x.begin(), prob_x.end());

    // thrust::device_vector <thrust::device_vector<float>> coreset(n_coreset);

    // for (int i = 0; i < n_coreset; i++){
    //     int id = d(gen);
    //     coreset[i] = device_points[id];
    // }

    return coreset;
}




