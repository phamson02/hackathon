#include <hip/hip_runtime.h>
#include <iostream>

__global__ void fused_moe_kernel(const float *__restrict__ hidden,
                                 const float *__restrict__ w1,
                                 const float *__restrict__ w2,
                                 const float *__restrict__ topk_weight,
                                 const int *__restrict__ topk_ids,
                                 float *__restrict__ output,
                                 int num_token,
                                 int model_dim,
                                 int inter_dim,
                                 int topk)
{
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    const int nThread = blockDim.x;
    const int stride = nThread;

    const float *h_vec = hidden + token * model_dim;

    for (int d = tid; d < model_dim; d += stride)
        output[token * model_dim + d] = 0.0f;

    extern __shared__ float smem[];
    float *act_input = smem;
    float *inter = smem + 2 * inter_dim;

    for (int k = 0; k < topk; ++k) {
        const int expert = topk_ids[token * topk + k];
        const float router_w = topk_weight[token * topk + k];

        const float *W1_e = w1 + ((size_t)expert) * (2 * inter_dim) * model_dim;

        for (int row = tid; row < 2 * inter_dim; row += stride) {
            const float *w_row = W1_e + (size_t)row * model_dim;
            float acc = 0.f;
            for (int d = 0; d < model_dim; ++d)
                acc += h_vec[d] * w_row[d];
            act_input[row] = acc;
        }
        __syncthreads();

        for (int j = tid; j < inter_dim; j += stride) {
            float gate = act_input[j];
            float up = act_input[inter_dim + j];
            float silu = gate / (1.f + __expf(-gate));
            inter[j] = silu * up;
        }
        __syncthreads();

        const float *W2_e = w2 + ((size_t)expert) * model_dim * inter_dim;

        for (int d = tid; d < model_dim; d += stride) {
            const float *w_row = W2_e + (size_t)d * inter_dim;
            float acc = 0.f;
            for (int j = 0; j < inter_dim; ++j)
                acc += w_row[j] * inter[j];

            output[token * model_dim + d] += router_w * acc;
        }
        __syncthreads();
    }
}

bool moe(int num_token,
         int model_dim,
         int inter_dim,
         int num_experts,
         int topk,
         float *hidden_states_cpu,
         float *w1_cpu,
         float *w2_cpu,
         float *topk_weight_cpu,
         int *topk_ids_cpu,
         float *output_cpu,
         float *hidden_states_gpu,
         float *w1_gpu,
         float *w2_gpu,
         float *topk_weight_gpu,
         int *topk_ids_gpu,
         float *output_gpu)
{
    const int threadsPerBlock = 256;
    dim3 block(threadsPerBlock);
    dim3 grid(num_token);

    std::size_t shm_bytes = static_cast<std::size_t>(3 * inter_dim) * sizeof(float);

    hipMemset(output_gpu, 0, static_cast<std::size_t>(num_token) * static_cast<std::size_t>(model_dim) * sizeof(float));

    hipLaunchKernelGGL(fused_moe_kernel,
                       grid, block, shm_bytes, 0,
                       hidden_states_gpu,
                       w1_gpu,
                       w2_gpu,
                       topk_weight_gpu,
                       topk_ids_gpu,
                       output_gpu,
                       num_token,
                       model_dim,
                       inter_dim,
                       topk);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
        std::cerr << "HIP launch failed: "
                  << hipGetErrorString(err) << std::endl;

    return true;
}