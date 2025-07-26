#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>

#define HIP_CHECK(cmd)                                                   \
    do {                                                                 \
        hipError_t e = (cmd);                                            \
        if (e != hipSuccess) {                                           \
            fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(e));                               \
            return false;                                                \
        }                                                                \
    } while (0)

static __device__ __forceinline__ int upper_bound_row(const int *__restrict__ A, int64_t row_start, int N, int x)
{
    int low = 0, high = N - 1;
    while (low < high) {
        int mid = (low + high) >> 1;
        int v = A[row_start + mid];
        if (v < x)
            low = mid + 1;
        else
            high = mid - 1;
    }
    return low;
}

static __device__ __forceinline__ int upper_bound_col(const int *__restrict__ A, int col, int N, int x)
{
    int low = 0, high = N - 1;
    while (low < high) {
        int mid = (low + high) >> 1;
        int v = A[(int64_t)mid * N + col];
        if (v < x)
            low = mid + 1;
        else
            high = mid - 1;
    }
    return low;
}

__global__ void kernel_parallel_bs(int N, int M,
                                   const int *__restrict__ A,
                                   const int *__restrict__ Q,
                                   int *__restrict__ out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M)
        return;

    int p = Q[2 * tid + 0];
    int x = Q[2 * tid + 1];

    if (p > 0) {
        int row = p - 1;
        int64_t base = (int64_t)row * N;
        out[tid] = upper_bound_row(A, base, N, x);
    }
    else {
        int col = (-p) - 1;
        out[tid] = upper_bound_col(A, col, N, x);
    }
}

bool psearch(int N, int M,
             int *A_cpu, int *Q_cpu, int *output_cpu,
             int *A_gpu, int *Q_gpu, int *output_gpu)
{
    const int BLOCK = 256;
    dim3 block(BLOCK);
    dim3 grid((M + BLOCK - 1) / BLOCK);

    hipLaunchKernelGGL(kernel_parallel_bs, grid, block, 0, 0,
                       N, M, A_gpu, Q_gpu, output_gpu);
    return true;
}