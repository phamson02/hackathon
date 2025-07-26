#include <iostream>
#include <chrono>
#include <vector>

#include <hip/hip_runtime.h>

using namespace std;

bool rmsnormq(int M, int N, int K, float* input_cpu, float* output_cpu, float* input_gpu, float* output_gpu);

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
    int M, N, K;
    cin >> M >> N >> K;

    float* input_gpu;
    float* output_gpu;

    vector<float> input_cpu(1ll * M * N);
    for (size_t i = 0; i < input_cpu.size(); ++i)
        input_cpu[i] = random_float(-1, 1);
    hipMalloc(&input_gpu, input_cpu.size() * sizeof(float));
    hipMemcpy(input_gpu, input_cpu.data(), input_cpu.size() * sizeof(float), hipMemcpyHostToDevice);

    vector<float> output_cpu(1ll * M * N);
    hipMalloc(&output_gpu, output_cpu.size() * sizeof(float));

    hipDeviceSynchronize();
    auto start = chrono::high_resolution_clock::now();
    bool use_gpu = rmsnormq(M, N, K, input_cpu.data(), output_cpu.data(), input_gpu, output_gpu);
    hipDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(end - start).count();

    // Copy output from GPU to CPU if needed
    if (use_gpu) {
        hipMemcpy(output_cpu.data(), output_gpu, output_cpu.size() * sizeof(float), hipMemcpyDeviceToHost);
        hipDeviceSynchronize();
    }

    // Print duration and output
    cout << duration << '\n';
    for (size_t i = 0; i < output_cpu.size(); ++i)
        cout << output_cpu[i] << ' ';
    cout << '\n';

    hipFree(input_gpu);
    hipFree(output_gpu);
    hipDeviceSynchronize();

    return 0;
}