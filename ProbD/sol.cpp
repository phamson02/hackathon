#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

using namespace std;

static constexpr long long MOD = 1'000'000'000LL;

#define HIP_CHECK(cmd)                                            \
    do {                                                          \
        hipError_t _e = (cmd);                                    \
        if (_e != hipSuccess) {                                   \
            cerr << "HIP error: " << hipGetErrorString(_e)        \
                 << " @ " << __FILE__ << ":" << __LINE__ << endl; \
            std::exit(1);                                         \
        }                                                         \
    } while (0)

// res[i] = sum_j a[(i + j) % N] * b[j]  (mod 1e9)
__global__ void cyclic_conv(const long long *__restrict__ a,
                            const long long *__restrict__ b,
                            long long *__restrict__ c,
                            int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;

    long long sum = 0;
    for (int j = 0; j < N; ++j) {
        int idx = i + j;
        if (idx >= N)
            idx -= N;
        long long prod = (a[idx] * b[j]) % MOD;
        sum += prod;
        if (sum >= MOD)
            sum -= MOD;
    }
    c[i] = sum;
}

bool findall(int N, long long K, int A, int B, int *output_cpu, int *output_gpu)
{
    if (K == 0) {
        output_cpu[0] = (A == B) ? 1 : 0;
        HIP_CHECK(hipMemcpy(output_gpu, output_cpu, sizeof(int), hipMemcpyHostToDevice));
        return true;
    }
    if (N == 1) {
        output_cpu[0] = 1;
        HIP_CHECK(hipMemcpy(output_gpu, output_cpu, sizeof(int), hipMemcpyHostToDevice));
        return true;
    }

    vector<long long> base(N, 0), result(N, 0);
    base[1 % N] = 1;
    base[(N - 1) % N] = 1;
    result[0] = 1;

    size_t bytes = static_cast<size_t>(N) * sizeof(long long);
    long long *d_res = nullptr, *d_pow = nullptr, *d_tmp = nullptr;
    HIP_CHECK(hipMalloc(&d_res, bytes));
    HIP_CHECK(hipMalloc(&d_pow, bytes));
    HIP_CHECK(hipMalloc(&d_tmp, bytes));

    HIP_CHECK(hipMemcpy(d_res, result.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_pow, base.data(), bytes, hipMemcpyHostToDevice));

    const int BLOCK = 256;
    const int GRID = (N + BLOCK - 1) / BLOCK;

    long long e = K;
    while (e > 0) {
        if (e & 1LL) {
            hipLaunchKernelGGL(cyclic_conv, dim3(GRID), dim3(BLOCK), 0, 0,
                               d_res, d_pow, d_tmp, N);
            HIP_CHECK(hipGetLastError());
            std::swap(d_res, d_tmp);
        }
        hipLaunchKernelGGL(cyclic_conv, dim3(GRID), dim3(BLOCK), 0, 0,
                           d_pow, d_pow, d_tmp, N);
        HIP_CHECK(hipGetLastError());
        std::swap(d_pow, d_tmp);
        e >>= 1;
    }

    int idx = ((B - A) % N + N) % N;
    long long ans;
    HIP_CHECK(hipMemcpy(&ans, d_res + idx, sizeof(long long), hipMemcpyDeviceToHost));
    output_cpu[0] = static_cast<int>(ans % MOD);

    HIP_CHECK(hipMemcpy(output_gpu, output_cpu, sizeof(int), hipMemcpyHostToDevice));

    HIP_CHECK(hipFree(d_res));
    HIP_CHECK(hipFree(d_pow));
    HIP_CHECK(hipFree(d_tmp));
    return true;
}