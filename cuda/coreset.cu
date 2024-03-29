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

 __global__ void vector_multiply(float* vector_a, float* vector_b, float* output, size_int n){
     int tid = blockIdx.x * blockDim.x + threadIdx.x;

     if (tid < n){
         output[tid] = vector_a[tid] * vector_b[tid];
     }

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

/*
__global__ void
_calculate_weights(float dist_sum, float* weights, float* dist,
                   size_int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n)
        weights[tid] = dist[tid] / dist_sum;
}*/

__global__ void
normalize(float* outputs, float* values, float values_sum,
		 size_int n) {
     int tid = blockIdx.x * blockDim.x + threadIdx.x;

     if (tid < n)
         outputs[tid] = values[tid] / values_sum;
}


//Change the function to accept 1-d form input
void k_means_pp_init_cu(float *points, float *data_weights, size_int size, float *centers, int n_cluster, unsigned int dimension) {
    if (n_cluster < 1) {
        throw "n_cluster, the number of clusters should at least greater than 1.";
    }
    if (n_cluster > size) {
        throw "n_cluster, the number of clusters should not greater than the size of whole points.";
    }

    //thrust::device_vector<float> centers(n_cluster * dimension); // 1 d
    //size_int size = points.size() / dimension;
    //Allocate CPU memory
    //vector<float> tmp_dist(size, FLOAT_MAX_VALUE);
    float dist[size];
    float weighted_dist[size];
    for (int i=0; i<size; i++){    // initialize the dist array with the max float value
        dist[i] = FLOAT_MAX_VALUE;
    }
    //copy(tmp_dist.begin(), tmp_dist.end(), dist);

    float weights[size];
    float weighted_dist_sum; // todo, try [long double] type
    //Allocate GPU memory
    
    float *d_dist, *d_weights, *d_points, *d_centers, *d_data_weights, *d_weighted_dist;

    CHECK(cudaMalloc((void**)&d_dist, sizeof(float) * size));
    CHECK(cudaMalloc((void**)&d_weighted_dist, sizeof(float) * size));
    CHECK(cudaMalloc((void**)&d_weights, sizeof(float) * size));
    CHECK(cudaMalloc((void**)&d_data_weights, sizeof(float) * size));
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
    CHECK(cudaMemcpy(d_data_weights, data_weights, sizeof(float) * size, cudaMemcpyHostToDevice));
    
    int cur_center_index;
    for (int i = 0; i < n_cluster-1; i++) { 
        weighted_dist_sum = 0.0;
        // launching kernel
        unsigned int center_start_id = i * dimension;
        
	int block_size = 256;
	int grid_size = divup(size, block_size);  // ensure that we call enough thread
	//Run kernel
        _min_dist<<<grid_size, block_size>>>(d_centers, center_start_id, d_points, d_dist, size, dimension);
        //Time the point weights with it's dist
        vector_multiply<<<grid_size, block_size>>>(d_dist, d_data_weights, d_weighted_dist, size);
	//copy d_dist back to CPU
	cudaMemcpy(weighted_dist, d_weighted_dist, size * sizeof(float), cudaMemcpyDeviceToHost);
	//Using thrust library to get the sum
	thrust::device_vector<float> device_dist(weighted_dist, weighted_dist + size);
        
	weighted_dist_sum = thrust::reduce(device_dist.begin(), device_dist.end(), (float) 0, thrust::plus<float>());
	// Run kernel
        normalize<<<grid_size, block_size>>>(d_weights, d_weighted_dist, weighted_dist_sum, size);
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
    CHECK(cudaFree(d_data_weights));
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
                                 float *cluster_weights,
                                 float dist_sum, size_int n) {


    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float res = device_weights[tid] * ((dist[tid] / dist_sum) + 1.0 / cluster_weights[assign[tid]]);
        sigma[tid] = res; 
    }
}

/*
__global__ void _compute_prob_x(float* sigma_x, float* prob_x, float sigma_sum, size_int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        prob_x[tid] = sigma_x[tid] / sigma_sum; 
    }
}
*/

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
    float weighted_dist[n];
    float sigma[n];
    unsigned int assign[n];
    float cluster_weights[n_cluster];
    float weighted_dist_sum; // todo, try [long double] type
    float sigma_sum;

    //Initialize cluster_size
    for(int i=0; i<n_cluster; i++){
        cluster_weights[i] = 0.0;
    }

    //Allocate GPU memory
    float *d_dist, *d_sigma, *d_prob_x, *d_points, *d_data_weights, *d_centers, *d_weights, *d_cluster_weights, *d_weighted_dist;
    unsigned int *d_assign;

    cudaMalloc((void**)&d_dist, sizeof(float) * n);
    cudaMalloc((void**)&d_sigma, sizeof(float) * n);
    cudaMalloc((void**)&d_points, sizeof(float) * n * dimension);
    cudaMalloc((void**)&d_data_weights, sizeof(float) * n);
    cudaMalloc((void**)&d_centers, sizeof(float) * n_cluster * dimension);
    cudaMalloc((void**)&d_prob_x, sizeof(float) * n);
    cudaMalloc((void**)&d_weights, sizeof(float) * n);
    cudaMalloc((void**)&d_weighted_dist, sizeof(float) * n);
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
        cluster_weights[assign[i]] += data_weights[i];
    }

    cudaMemcpy(d_cluster_weights, cluster_weights, n_cluster * sizeof(float), cudaMemcpyHostToDevice);
    
    //Compute the weighted dist
    vector_multiply<<<grid_size, block_size>>>(d_data_weights, d_dist, d_weighted_dist, n);
    cudaMemcpy(weighted_dist, d_weighted_dist, n * sizeof(float), cudaMemcpyDeviceToHost);

    //Using thrust library to get the sum of weighted dist
    thrust::device_vector<float> device_dist(weighted_dist, weighted_dist + n);
    weighted_dist_sum = thrust::reduce(device_dist.begin(), device_dist.end(), (float) 0, thrust::plus<float>());
    
    _compute_sigma_x<<<grid_size, block_size>>>(d_dist, d_data_weights, d_sigma, d_assign, d_cluster_weights, weighted_dist_sum, n);
    
    cudaMemcpy(sigma, d_sigma, n * sizeof(float), cudaMemcpyDeviceToHost);

    //Using thrust library to get the sum of sigma
    thrust::device_vector<float> device_sigma(sigma, sigma + n);
    sigma_sum = thrust::reduce(device_sigma.begin(), device_sigma.end(), (float) 0, thrust::plus<float>());
    normalize<<<grid_size, block_size>>>(d_prob_x, d_sigma, sigma_sum, n);  
    
    _compute_weights<<<grid_size, block_size>>>(d_prob_x, d_data_weights, d_weights, n, n_coreset);
    //copy d_weights back to CPU
    cudaMemcpy(prob_x, d_prob_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(weights, d_weights, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    //Free GPU memory
    CHECK(cudaFree(d_points));
    CHECK(cudaFree(d_centers));
    CHECK(cudaFree(d_dist)); 
    CHECK(cudaFree(d_weighted_dist));
    CHECK(cudaFree(d_prob_x));
    CHECK(cudaFree(d_sigma));
    CHECK(cudaFree(d_cluster_weights));
    CHECK(cudaFree(d_assign));
    CHECK(cudaFree(d_weights));

}



// Compute Coreset Function, return the Points type 
coreset::Points
compute_coreset(vector<float> &points, vector<float> &data_weights, unsigned int dimension, unsigned int n_cluster, size_int n_coreset) {
    size_int n = points.size() / dimension;
    size_int data_size = points.size();

    if (n < n_coreset) {
        throw "Setting size of coreset is greater or equal to the original data size, please alter it";
    }

    coreset::Points coreset(n_coreset, dimension); //define coreset object

    float host_points[data_size];
    copy(points.begin(), points.end(), host_points);
    float host_weights[n];
    copy(data_weights.begin(), data_weights.end(), host_weights);
    
    float centers[n_cluster * dimension];
    k_means_pp_init_cu(host_points, host_weights, n,centers, n_cluster, dimension);
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


// Compute coreset function, return the FlatPoints object
coreset::FlatPoints
compute_coreset_flat(vector<float> &points, vector<float> &data_weights, unsigned int dimension, unsigned int n_cluster, size_int n_coreset) {
    size_int n = points.size() / dimension;
    size_int data_size = points.size();

    if (n < n_coreset) {
        throw "Setting size of coreset is greater or equal to the original data size, please alter it";
    }

    float host_points[data_size];
    copy(points.begin(), points.end(), host_points);
    float host_weights[n];
    copy(data_weights.begin(), data_weights.end(), host_weights);

    float centers[n_cluster * dimension];
    k_means_pp_init_cu(host_points, host_weights, n,centers, n_cluster, dimension);
    //thrust::device_vector<float> prob_x;
    float prob_x[n];
    float weights[n];

    _compute_sigma(host_points, host_weights, centers, n, prob_x, weights, n_cluster, dimension, n_coreset);

    size_int sample_idx[n_coreset];

    random_weight_sample_cuda(n_coreset, sample_idx, prob_x, n); // sample coreset

    //select the samples and weights by samples index
    vector<float> samples(n_coreset * dimension);
    vector<float> samples_weights(n_coreset);
    
    size_int ind = 0;
    for(int i=0; i<n_coreset; i++){
        size_int sid = sample_idx[i];
        size_int data_start_id = sid * dimension;

        for(int j=0; j<dimension; j++){
                samples[ind] = host_points[data_start_id + j];
		ind ++;
        }

        samples_weights[i]=weights[sid];
    }
    //Using Point object to store the coreset
    coreset::FlatPoints coreset(n_coreset, dimension); //define coreset object
    coreset.FillPoints(samples, samples_weights);
    return coreset;
}
