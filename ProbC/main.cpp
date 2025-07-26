#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>

#include <hip/hip_runtime.h>

using namespace std;

bool psearch(int N, int M, int* A_cpu, int* Q_cpu, int* output_cpu, int* A_gpu, int* Q_gpu, int* output_gpu);

uint64_t state = 2025;
uint64_t fast_random(uint64_t& state) {
    state = (state * 48271) % 2147483647;
    return state;
}
int random_int(int min, int max) {
    return min + (fast_random(state) % (max - min + 1));
}
float random_float(float min, float max) {
    return min + (fast_random(state) % (int)((max - min) * 1000)) / 1000.0f;
}

int main() {
    int N, M;
    cin >> N >> M;

    int* A_gpu;
    int* Q_gpu;
    int* output_gpu;

    vector<int> A_cpu(1ll * N * N);
    for (size_t i = 0; i < A_cpu.size(); ++i)
        A_cpu[i] = (i < N ? i : A_cpu[i-N] + 1);
    hipMalloc(&A_gpu, A_cpu.size() * sizeof(int));
    hipMemcpy(A_gpu, A_cpu.data(), A_cpu.size() * sizeof(int), hipMemcpyHostToDevice);

    vector<int> Q_cpu(M * 2);
    for (size_t i = 0; i < Q_cpu.size(); i+=2) {
        Q_cpu[i] = random_int(1, N) * (random_int(0, 1) ? 1 : -1);
        if (Q_cpu[i] > 0)
            Q_cpu[i + 1] = Q_cpu[i] - 1 + random_int(0, N);
        else
            Q_cpu[i + 1] = -Q_cpu[i] - 1 + random_int(0, N);
    }
    hipMalloc(&Q_gpu, Q_cpu.size() * sizeof(int));
    hipMemcpy(Q_gpu, Q_cpu.data(), Q_cpu.size() * sizeof(int), hipMemcpyHostToDevice);

    vector<int> output_cpu(M);
    hipMalloc(&output_gpu, output_cpu.size() * sizeof(int));

    hipDeviceSynchronize();
    auto start = chrono::high_resolution_clock::now();
    bool use_gpu = psearch(N, M, A_cpu.data(), Q_cpu.data(), output_cpu.data(), A_gpu, Q_gpu, output_gpu);
    hipDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(end - start).count();

    // Copy output from GPU to CPU if needed
    if (use_gpu) {
        hipMemcpy(output_cpu.data(), output_gpu, output_cpu.size() * sizeof(int), hipMemcpyDeviceToHost);
        hipDeviceSynchronize();
    }

    // Print duration and output
    cout << duration << '\n';
    for (size_t i = 0; i < output_cpu.size(); ++i)
        cout << output_cpu[i] << ' ';
    cout << '\n';

    hipFree(A_gpu);
    hipFree(Q_gpu);
    hipFree(output_gpu);
    hipDeviceSynchronize();

    return 0;
}