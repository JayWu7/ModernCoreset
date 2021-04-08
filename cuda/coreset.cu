// Created by Xiaobo Wu on 2021/2/1.
//
#include <thrust/host_vector.h>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>
#include "points.h"
#include "points.cpp"
#include "random_sample.cu"
typedef unsigned long long int size_int;
#define FLOAT_MAX_VALUE 3.40282e+038
using namespace std;


/*__device__ float _squared_l2_distance(thrust::device_vector<float> &first, thrust::device_vector<float> &second, unsigned int dimension) {
    float distance = 0.0;
    //int dimension = first.size();
    //if (dimension != second.size()) {
       // throw "Input two vector are not in the same dimension!";
    //}  device code does not support exception handling
    for (int i = 0; i < dimension; i++) {
        float gap = first[i] - second[i];
        distance += gap * gap;
    }
    return distance;
}*/


static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)


__device__ float _squared_l2_distance_with_index(float* first, size_int fid, float* second, size_int sid, unsigned int dimension) {
    float distance = 0.0;
    for (int i = 0; i < dimension; i++) {
        float gap = first[fid + i] - second[sid + i];
        distance += gap * gap;
    }
    return distance;
}

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

//todo, test if using [long double] type of dist_sum will influence the performance
__global__ void _min_dist(float* centers, unsigned int center_id, 
                          float* device_points,
                          float* dist, size_int n, unsigned int dimension) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        int sid = tid * dimension;
        float new_dis = _squared_l2_distance_with_index(centers, center_id, device_points, sid, dimension);
        if (new_dis < dist[tid]) {
            dist[tid] = new_dis;
        }
    }
}


__global__ void
_calculate_weights(float dist_sum, float* weights, float* dist,
                   size_int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n)
        weights[tid] = dist[tid] / dist_sum;
}


//Change the function to accept 1-d form input
void k_means_pp_init_cu(float *points, size_int size, float *centers, int n_cluster, unsigned int dimension) {
    if (n_cluster < 1) {
        throw "n_cluster, the number of clusters should at least greater than 1.";
    }

    //thrust::device_vector<float> centers(n_cluster * dimension); // 1 d
    //size_int size = points.size() / dimension;
    //Allocate CPU memory
    vector<float> tmp_dist(size, FLOAT_MAX_VALUE);
    float dist[size];
    copy(tmp_dist.begin(), tmp_dist.end(), dist);

    float weights[size];
    float dist_sum; // todo, try [long double] type
    //Allocate GPU memory
    
    float *d_dist, *d_weights, *d_points, *d_centers;

    CHECK(cudaMalloc((void**)&d_dist, sizeof(float) * size));
    CHECK(cudaMalloc((void**)&d_weights, sizeof(float) * size));
    CHECK(cudaMalloc((void**)&d_points, sizeof(float) * size * dimension));
    CHECK(cudaMalloc((void**)&d_centers, sizeof(float) * n_cluster * dimension));
    
    //Choose one center uniformly at random among the data points.
    random_device rd;  //Will be used to obtain a seed for the random number engine
    mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    uniform_int_distribution<> distrib(0, size - 1);
    int first_center_index = distrib(gen);


    int start_index = first_center_index * dimension;
    for (int i = 0; i < dimension; i++){
        centers[i] = points[start_index + i];
    }


    //Copy data
    CHECK(cudaMemcpy(d_points, points, sizeof(float) * size * dimension, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_centers, centers, sizeof(float) * n_cluster * dimension, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dist, dist, sizeof(float) * size, cudaMemcpyHostToDevice));
    
    int cur_center_index;
    for (int i = 0; i < n_cluster-1; i++) { 
        dist_sum = 0.0;
        // launching kernel
        unsigned int center_start_id = i * dimension;
        
	int block_size = 256;
	int grid_size = divup(size, block_size);  // ensure that we call enough thread
	//Run kernel
        _min_dist<<<grid_size, block_size>>>(d_centers, center_start_id, d_points, d_dist, size, dimension);
        
	//copy d_dist back to CPU
	cudaMemcpy(dist, d_dist, size * sizeof(float), cudaMemcpyDeviceToHost);
	//Using thrust library to get the sum
	thrust::device_vector<float> device_dist(dist, dist + size);
        
	dist_sum = thrust::reduce(device_dist.begin(), device_dist.end(), (float) 0, thrust::plus<float>());
	// Run kernel
        _calculate_weights<<<grid_size, block_size>>>(dist_sum, d_weights, d_dist, size);
	//copy d_weights back to CPU
	
	CHECK(cudaMemcpy(weights, d_weights, size * sizeof(float), cudaMemcpyDeviceToHost));
	discrete_distribution<int> dd(weights, weights + size);
        cur_center_index = dd(gen);

        start_index = cur_center_index * dimension;
        center_start_id += dimension; 
        for (int i = 0; i < dimension; i++){
            centers[center_start_id + i] = points[start_index + i];
        }
	CHECK(cudaMemcpy(d_centers, centers, sizeof(float) * n_cluster * dimension, cudaMemcpyHostToDevice));
    }
    //Free memory
    CHECK(cudaFree(d_points));
    CHECK(cudaFree(d_centers));
    CHECK(cudaFree(d_weights));
    CHECK(cudaFree(d_dist));
}


//todo, test if using [long double] type of dist_sum will influence the performance
__global__ void _compute_sigma_dist(float* device_points,
                                    float* centers,
                                    float* dist, unsigned int* assign,
                                    size_int n, 
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
        assign[tid] = center_id;
    }
}


__global__ void _compute_sigma_x(float* dist, float* device_weights, float* sigma,
                                 unsigned int* assign,
                                 size_int *cluster_size, float *cluster_weights,
                                 float dist_sum, size_int n) {


    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float res = device_weights[tid] * ((dist[tid] / dist_sum) + 1.0 / (cluster_weights[assign[n]] * cluster_size[assign[tid]]));
        sigma[tid] = res; 
    }
}


__global__ void _compute_prob_x(float* sigma_x, float* prob_x, float sigma_sum, size_int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        prob_x[tid] = sigma_x[tid] / sigma_sum; 
    }
}


__global__ void _compute_weights(float* prob_x, float* device_weights, float* weight_x, size_int n, size_int n_coreset) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        weight_x[tid] = device_weights[tid] / (n_coreset * prob_x[tid]); 
    }
}


void
_compute_sigma(float* points,
               float* data_weights,
               float* centers, size_int n, 
	       float* prob_x, float* weights,
               unsigned int n_cluster, unsigned int dimension, size_int n_coreset) {
   
    //Allocate CPU memory
    float dist[n];
    float sigma[n];
    unsigned int assign[n];
    size_int cluster_size[n_cluster];
    float cluster_weights[n_cluster];
    float dist_sum; // todo, try [long double] type
    float sigma_sum;

    //Initialize cluster_size
    for(int i=0; i<n_cluster; i++){
        cluster_size[i] = 0;
        cluster_weights[i] = 0.0;
    }

    //Allocate GPU memory
    float *d_dist, *d_sigma, *d_prob_x, *d_points, *d_data_weights, *d_centers, *d_weights, *d_cluster_weights;
    size_int *d_cluster_size;
    unsigned int *d_assign;

    cudaMalloc((void**)&d_dist, sizeof(float) * n);
    cudaMalloc((void**)&d_sigma, sizeof(float) * n);
    cudaMalloc((void**)&d_points, sizeof(float) * n * dimension);
    cudaMalloc((void**)&d_data_weights, sizeof(float) * n);
    cudaMalloc((void**)&d_centers, sizeof(float) * n_cluster * dimension);
    cudaMalloc((void**)&d_prob_x, sizeof(float) * n);
    cudaMalloc((void**)&d_weights, sizeof(float) * n);
    cudaMalloc((void**)&d_cluster_size, sizeof(size_int) * n_cluster);
    cudaMalloc((void**)&d_cluster_weights, sizeof(float) * n_cluster);
    cudaMalloc((void**)&d_assign, sizeof(unsigned int) * n);

    //Copy data
    cudaMemcpy(d_points, points, sizeof(float) * n * dimension, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_weights, data_weights, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centers, centers, sizeof(float) * n_cluster * dimension, cudaMemcpyHostToDevice);
    //Launch kernel
    int block_size = 256; // in each block, we have 256 threads
    int grid_size = divup(n, block_size);  // ensure that we call enough block

    _compute_sigma_dist<<<grid_size, block_size>>>(d_points, d_centers, d_dist, d_assign, n, n_cluster, dimension);
    
    cudaMemcpy(dist, d_dist, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(assign, d_assign, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    for(int i=0; i<n; i++){
    	cluster_size[assign[i]] += 1;
        cluster_weights[assign[i]] += data_weights[i];
    }

    cudaMemcpy(d_cluster_size, cluster_size, n_cluster * sizeof(size_int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_weights, cluster_weights, n_cluster * sizeof(float), cudaMemcpyHostToDevice);

    //Using thrust library to get the sum of dist
    thrust::device_vector<float> device_dist(dist, dist + n);
    dist_sum = thrust::reduce(device_dist.begin(), device_dist.end(), (float) 0, thrust::plus<float>());

    _compute_sigma_x<<<grid_size, block_size>>>(d_dist, d_data_weights, d_sigma, d_assign, d_cluster_size, d_cluster_weights, dist_sum, n);
    
    cudaMemcpy(sigma, d_sigma, n * sizeof(float), cudaMemcpyDeviceToHost);

    //Using thrust library to get the sum of sigma
    thrust::device_vector<float> device_sigma(sigma, sigma + n);
    sigma_sum = thrust::reduce(device_sigma.begin(), device_sigma.end(), (float) 0, thrust::plus<float>());
    _compute_prob_x<<<grid_size, block_size>>>(d_sigma, d_prob_x, sigma_sum, n);  
    
    _compute_weights<<<grid_size, block_size>>>(d_prob_x, d_data_weights, d_weights, n, n_coreset);
    //copy d_weights back to CPU
    cudaMemcpy(prob_x, d_prob_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(weights, d_weights, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    //Free GPU memory
    CHECK(cudaFree(d_points));
    CHECK(cudaFree(d_centers));
    CHECK(cudaFree(d_dist)); 
    CHECK(cudaFree(d_prob_x));
    CHECK(cudaFree(d_sigma));
    CHECK(cudaFree(d_cluster_size));
    CHECK(cudaFree(d_cluster_weights));
    CHECK(cudaFree(d_assign));
    CHECK(cudaFree(d_weights));

}



// Compute Coreset Function, return the Points type 
coreset::Points
compute_coreset(vector<float> &points, vector<float> &data_weights, unsigned int dimension, unsigned int n_cluster, size_int n_coreset) {
    size_int n = points.size() / dimension;
    size_int data_size = points.size();

    if (data_size < n_coreset) {
        throw "Setting size of coreset is greater or equal to the original data size, please alter it";
    }

    coreset::Points coreset(n_coreset, dimension); //define coreset class

    float host_points[data_size];
    copy(points.begin(), points.end(), host_points);
    float host_weights[n];
    copy(data_weights.begin(), data_weights.end(), host_weights);
    
    float centers[n_cluster * dimension];
    k_means_pp_init_cu(host_points, n,centers, n_cluster, dimension);

    //thrust::device_vector<float> prob_x;
    float prob_x[n];
    float weights[n];
    
    _compute_sigma(host_points, host_weights, centers, n, prob_x, weights, n_cluster, dimension, n_coreset);
   
    size_int sample_idx[n_coreset];

    random_weight_sample_cuda(n_coreset, sample_idx, prob_x, n); // sample coreset
    
    //select the samples and weights by samples index
    vector<vector<float> > samples(n_coreset);
    vector<float> samples_weights(n_coreset);

    for(int i=0; i<n_coreset; i++){
        vector<float> sample(dimension);
        size_int sid = sample_idx[i];
        size_int data_start_id = sid * dimension;

        for(int j=0; j<dimension; j++){
                sample[j] = host_points[data_start_id + j];
            }

        samples[i] = sample;
        samples_weights[i]=weights[sid];
    }
    //Using Point object to store the coreset
    coreset.FillPoints(samples, samples_weights);
    return coreset;
}

