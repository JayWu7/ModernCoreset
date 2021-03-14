#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <thrust/scan.h>


#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) \     ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 
typedef unsigned long long int size_int;

using namespace std;

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {

    /* we have to initialize the state */
    curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
                blockIdx.x, /* the sequence number should be different for each core (unless you want all
                               cores to get the same sequence of numbers for some reason - use thread id! */
                0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &states[blockIdx.x]);
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void randoms(curandState_t* states, float* numbers) {
    /* curand works like rand - except that it takes a state as a parameter */
    numbers[blockIdx.x] = curand_uniform(&states[blockIdx.x]);
}


__global__ void binary_search_id(size_int *sample_idx, float *numbers, float *prefix_sum, unsigned int N, size_int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int l = 0;
        int r = n - 1; 
        float k = numbers[tid];
        int mid;

        while (l < r){
            mid = (l + r) / 2;
            if(prefix_sum[mid] < k)
                l = mid + 1;
            else
                r = mid;
        }
        sample_idx[tid] = r; 
    }
}


/*
void random_generator(unsigned int N, float *cpu_nums)
{
     //CUDA's random number library uses curandState_t to keep track of the seed value        we will store a random state for every thread
     curandState_t* states;
     // allocate space on the GPU for the random states
     cudaMalloc((void**) &states, N * sizeof(curandState_t));
     // invoke the GPU to initialize all of the random states
     init<<<N, 1>>>(time(0), states);
     // allocate an array of unsigned ints on the CPU and GPU
     float* gpu_nums;
     cudaMalloc((void**) &gpu_nums, N * sizeof(float));

     // invoke the kernel to get some random numbers
     randoms<<<N, 1>>>(states, gpu_nums, 100);

     // copy the random numbers back
     cudaMemcpy(cpu_nums, gpu_nums, N * sizeof(float), cudaMemcpyDeviceToHost);

     // free the memory we allocated for the states and numbers 
     cudaFree(states);
     cudaFree(gpu_nums);
}
*/

void random_weight_sample_cuda(unsigned int N, size_int *sample_idx, float *weights, size_int n){
    //Compute the prefix sum of weights
    thrust::inclusive_scan(weights, weights + n, weights); // in-place scan

    // Generate N random numbers, between (0,1]
    curandState_t* states;
    /* allocate space on the GPU for the random states */
    cudaMalloc((void**) &states, N * sizeof(curandState_t));
    /* invoke the GPU to initialize all of the random states */
    init<<<N, 1>>>(time(0), states);
    /* allocate an array of unsigned ints on the CPU and GPU */
    float* gpu_nums;
    cudaMalloc((void**) &gpu_nums, N * sizeof(float));

    /* invoke the kernel to get some random numbers */
    randoms<<<N, 1>>>(states, gpu_nums);
    
    //allocate gpu array for d_weights and d_sample_idx
    float* d_weights;
    cudaMalloc((void**) &d_weights, n * sizeof(float));
    size_int* d_sample_idx;
    cudaMalloc((void**) &d_sample_idx, N * sizeof(size_int));

    //copy weights array to d_weights
    cudaMemcpy(d_weights, weights, sizeof(float) * n, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size - 1)/block_size;  // ensure that we call enough thread
    binary_search_id<<<grid_size, block_size>>>(d_sample_idx, gpu_nums, d_weights, N, n);
    
    //copy d_sample_idx back to CPU
    cudaMemcpy(sample_idx, d_sample_idx, N * sizeof(size_int), cudaMemcpyDeviceToHost);

     /* free the memory we allocated for the states and numbers */
    cudaFree(states);
    cudaFree(gpu_nums);
    cudaFree(d_weights);
    cudaFree(d_sample_idx);
 }

