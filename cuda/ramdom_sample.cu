#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) \     ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 

__global__ void prescan(float *g_odata, float *g_idata, int n) { 
    extern __shared__ float temp[];  // allocated on invocation 
    int thid = threadIdx.x; int offset = 1; 

    int ai = thid;
    int bi = thid + (n/2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    temp[ai + bankOffsetA] = g_idata[ai];
    temp[bi + bankOffsetB] = g_idata[bi];

    for (int d = n>>1; d > 0; d >>= 1){     // build sum in place up the tree
        __syncthreads();
        if (thid < d){
            int ai = offset*(2*thid+1)-1; 
            int bi = offset*(2*thid+2)-1;
            ai += CONFLICT_FREE_OFFSET(ai); 
            bi += CONFLICT_FREE_OFFSET(bi);
            temp[bi] += temp[ai];
        }
        offset *= 2; 
    }
    if (thid == 0) {   // clear the last element  
        temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
    }

    for (int d = 1; d < n; d *= 2) { // traverse down tree & build scan
        offset >>= 1;      
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1; 
            int bi = offset*(2*thid+2)-1;
            ai += CONFLICT_FREE_OFFSET(ai); 
            bi += CONFLICT_FREE_OFFSET(bi);
            float t = temp[ai];
            temp[ai] = temp[bi]; 
            temp[bi] += t;
        }
    }

    __syncthreads();
    g_odata[ai] = temp[ai + bankOffsetA]; 
    g_odata[bi] = temp[bi + bankOffsetB]; 
}


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
__global__ void randoms(curandState_t* states, unsigned int* numbers, unsigned int max_number) {
    /* curand works like rand - except that it takes a state as a parameter */
    numbers[blockIdx.x] = curand(&states[blockIdx.x]) % max_number;
}


__global__ void binary_search(unsigned int *numbers, float *prefix_sum, unsigned int N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int l = 0;
        int r = N;  // We add a zero at the begining of prefix_sum
        unsigned int k = numbers[tid];
        int mid;

        while (l + 1 < r){
            mid = (l + r) / 2;
            if(prefix_sum[mid] > k)
                r = mid;
            else if(prefix_sum[mid] < k)
                l = mid;
            else{
                // l = mid;
                r = mid;
                break;
            }
        }

        numbers[tid] = r - 1; // minus 1 to get the correct index in the original array(we add one item in the prefix_sum)
    }
}


void random_generator(unsigned int N, unsigned int max_number, unsigned int *numbers){
    /*
    Function to generate N random numbers
    */
    /* CUDA's random number library uses curandState_t to keep track of the seed value
       we will store a random state for every thread  */
       curandState_t* states;

       /* allocate space on the GPU for the random states */
       cudaMalloc((void**) &states, N * sizeof(curandState_t));
   
       /* invoke the GPU to initialize all of the random states */
       init<<<N, 1>>>(time(0), states);
   
       /* allocate an array of unsigned ints on the CPU and GPU */
    //    unsigned int cpu_nums[N];
       unsigned int* gpu_nums;
       cudaMalloc((void**) &gpu_nums, N * sizeof(unsigned int));
   
       /* invoke the kernel to get some random numbers */
       randoms<<<N, 1>>>(states, gpu_nums, max_number);
   
       /* copy the random numbers back */
       cudaMemcpy(numbers, gpu_nums, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
   
   
       /* free the memory we allocated for the states and numbers */
       cudaFree(states);
       cudaFree(gpu_nums);
}

void random_weight_sample_cuda(unsigned int N, float *weights){
    // todo
    
}


// int main() {
//     /* CUDA's random number library uses curandState_t to keep track of the seed value
//        we will store a random state for every thread  */
//     curandState_t* states;

//     /* allocate space on the GPU for the random states */
//     cudaMalloc((void**) &states, N * sizeof(curandState_t));

//     /* invoke the GPU to initialize all of the random states */
//     init<<<N, 1>>>(time(0), states);

//     /* allocate an array of unsigned ints on the CPU and GPU */
//     unsigned int cpu_nums[N];
//     unsigned int* gpu_nums;
//     cudaMalloc((void**) &gpu_nums, N * sizeof(unsigned int));

//     /* invoke the kernel to get some random numbers */
//     randoms<<<N, 1>>>(states, gpu_nums);

//     /* copy the random numbers back */
//     cudaMemcpy(cpu_nums, gpu_nums, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);

//     /* print them out */
//     for (int i = 0; i < N; i++) {
//         printf("%u\n", cpu_nums[i]);
//     }

//     /* free the memory we allocated for the states and numbers */
//     cudaFree(states);
//     cudaFree(gpu_nums);

//     return 0;
// }


