#include <hip/hip_runtime.h>
#include <vector>
#include <algorithm>

#define MOD 1000
#define MAX_K 1024

__global__ void knapsack_shared_kernel(int N, int K, int* w, int* result) {
    __shared__ int dp[MAX_K + 1];
    int tid = threadIdx.x;

    if (tid <= K) {
        dp[tid] = (tid == 0) ? 1 : 0;
    }
    __syncthreads();

    for (int i = 0; i < N; ++i) {
        int wi = w[i];
        __syncthreads();
        int temp = (tid >= wi && tid <= K) ? dp[tid - wi] : 0;
        __syncthreads();
        if (tid >= wi && tid <= K) {
            dp[tid] = (dp[tid] + temp) % MOD;
        }

        __syncthreads();
    }

    if (tid == 0) {
        int total = 0;
        for (int i = 0; i <= K; ++i)
            total = (total + dp[i]) % MOD;
        result[0] = total;
    }
}

__global__ void knapsack_global_kernel(int K, int weight, int* dp_old, int* dp_new) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > K) return;

    if (tid >= weight)
        dp_new[tid] = (dp_old[tid] + dp_old[tid - weight]) % MOD;
    else
        dp_new[tid] = dp_old[tid];
}

bool predict(int N, int K, int* w_cpu, int* output_cpu, int* w_gpu, int* output_gpu) {
    if (K <= MAX_K) {
        int* result_gpu;
        hipMalloc(&result_gpu, sizeof(int));

        int threads = K + 1;
        knapsack_shared_kernel<<<1, threads>>>(N, K, w_gpu, result_gpu);

        hipMemcpy(output_cpu, result_gpu, sizeof(int), hipMemcpyDeviceToHost);
        hipMemcpy(output_gpu, output_cpu, sizeof(int), hipMemcpyHostToDevice);
        hipFree(result_gpu);
        return true;
    }

    int* dp1_gpu;
    int* dp2_gpu;

    std::vector<int> dp_init(K + 1, 0);
    dp_init[0] = 1;

    hipMalloc(&dp1_gpu, (K + 1) * sizeof(int));
    hipMalloc(&dp2_gpu, (K + 1) * sizeof(int));
    hipMemcpy(dp1_gpu, dp_init.data(), (K + 1) * sizeof(int), hipMemcpyHostToDevice);

    int threads = 256;
    int blocks = (K + threads) / threads;

    for (int i = 0; i < N; ++i) {
        int weight = w_cpu[i];  
        knapsack_global_kernel<<<blocks, threads>>>(K, weight, dp1_gpu, dp2_gpu);
        std::swap(dp1_gpu, dp2_gpu);
    }

    std::vector<int> dp_cpu(K + 1);
    hipMemcpy(dp_cpu.data(), dp1_gpu, (K + 1) * sizeof(int), hipMemcpyDeviceToHost);

    int total = 0;
    for (int i = 0; i <= K; ++i)
        total = (total + dp_cpu[i]) % MOD;

    output_cpu[0] = total;
    hipMemcpy(output_gpu, output_cpu, sizeof(int), hipMemcpyHostToDevice);

    hipFree(dp1_gpu);
    hipFree(dp2_gpu);
    return true;
}