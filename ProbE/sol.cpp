#include <hip/hip_runtime.h>
#include <cmath>

#define EPS 1e-6f

__device__ inline float warp_reduce_sum(float v){
    for (int off = warpSize >> 1; off > 0; off >>= 1)
        v += __shfl_down(v, off, warpSize);
    return v;
}
__device__ inline float warp_reduce_max(float v){
    for (int off = warpSize >> 1; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down(v, off, warpSize));
    return v;
}

__global__ void rmsnorm_group_kernel(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     int N, int K)
{
    int row = blockIdx.x;
    const float* row_in  = in  + row * N;
    float*       row_out = out + row * N;

    // 1) RMSNorm
    float local = 0.f;
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
        float v = row_in[c];
        local += v * v;
    }

    local = warp_reduce_sum(local);

    __shared__ float smem[32];
    int warp = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    if (lane == 0) smem[warp] = local;
    __syncthreads();

    float block_sum = 0.f;
    if (warp == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        block_sum = (lane < num_warps) ? smem[lane] : 0.f;
        block_sum = warp_reduce_sum(block_sum);
    }
    if (warp == 0 && lane == 0) smem[0] = block_sum;
    __syncthreads();
    block_sum = smem[0];

    float inv_rms = rsqrtf(block_sum / (float)N + EPS);

    for (int c = threadIdx.x; c < N; c += blockDim.x)
        row_out[c] = row_in[c] * inv_rms;
    __syncthreads();

    // 2) Group scaling
    int num_groups = (N + K - 1) / K;
    for (int g = 0; g < num_groups; ++g) {
        int start = g * K;
        int end   = (start + K < N) ? (start + K) : N;

        float lmax = 0.f;
        for (int c = start + threadIdx.x; c < end; c += blockDim.x)
            lmax = fmaxf(lmax, fabsf(row_out[c]));

        lmax = warp_reduce_max(lmax);
        if (lane == 0) smem[warp] = lmax;
        __syncthreads();

        float gmax = 0.f;
        if (warp == 0) {
            int num_warps = (blockDim.x + warpSize - 1) / warpSize;
            gmax = (lane < num_warps) ? smem[lane] : 0.f;
            gmax = warp_reduce_max(gmax);
        }
        if (warp == 0 && lane == 0) smem[0] = gmax;
        __syncthreads();
        gmax = smem[0];
        gmax = (gmax > 0.f) ? gmax : 1.f;

        for (int c = start + threadIdx.x; c < end; c += blockDim.x)
            row_out[c] /= gmax;
        __syncthreads();
    }
}

bool rmsnormq(int M, int N, int K,
              float* input_cpu, float* output_cpu,
              float* input_gpu, float* output_gpu)
{
    const int threads = 256;
    dim3 grid(M), block(threads);

    auto HIP_OK = [](hipError_t e){ return e == hipSuccess; };

    bool free_in=false, free_out=false;
    if (!input_gpu){ if(!HIP_OK(hipMalloc(&input_gpu,  M*N*sizeof(float)))) return false; free_in=true; }
    if (!output_gpu){ if(!HIP_OK(hipMalloc(&output_gpu, M*N*sizeof(float)))) return false; free_out=true; }

    if (input_cpu)
        if(!HIP_OK(hipMemcpy(input_gpu, input_cpu, M*N*sizeof(float), hipMemcpyHostToDevice))) return false;

    hipLaunchKernelGGL(rmsnorm_group_kernel, grid, block, 0, 0, input_gpu, output_gpu, N, K);
    if(!HIP_OK(hipGetLastError())) return false;

    if (output_cpu)
        if(!HIP_OK(hipMemcpy(output_cpu, output_gpu, M*N*sizeof(float), hipMemcpyDeviceToHost))) return false;

    if (free_in)  hipFree(input_gpu);
    if (free_out) hipFree(output_gpu);
    return true;
}
