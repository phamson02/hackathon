#include <iostream>
#include <chrono>
#include <vector>

#include <hip/hip_runtime.h>

using namespace std;

bool predict(int N, int K, int* w_cpu, int* output_cpu, int* w_gpu, int* output_gpu);

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
    int N, K;
    cin >> N >> K;

    int* w_gpu;
    int* output_gpu;

    vector<int> w_cpu(N);
    for (size_t i = 0; i < w_cpu.size(); ++i)
        w_cpu[i] = random_int(1, min(K, 100));
    hipMalloc(&w_gpu, w_cpu.size() * sizeof(int));
    hipMemcpy(w_gpu, w_cpu.data(), w_cpu.size() * sizeof(int), hipMemcpyHostToDevice);
    
    vector<int> output_cpu(1);
    hipMalloc(&output_gpu, output_cpu.size() * sizeof(int));

    hipDeviceSynchronize();
    auto start = chrono::high_resolution_clock::now();
    bool use_gpu = predict(N, K, w_cpu.data(), output_cpu.data(), w_gpu, output_gpu);
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

    hipFree(w_gpu);
    hipFree(output_gpu);
    hipDeviceSynchronize();

    return 0;
}