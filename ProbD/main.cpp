#include <iostream>
#include <chrono>
#include <vector>

#include <hip/hip_runtime.h>

using namespace std;

bool findall(int N, int K, int A, int B, int* output_cpu, int* output_gpu);

int main() {
    int N, K, A, B;
    cin >> N >> K >> A >> B;

    int* output_gpu;

    vector<int> output_cpu(1);
    hipMalloc(&output_gpu, output_cpu.size() * sizeof(int));

    hipDeviceSynchronize();
    auto start = chrono::high_resolution_clock::now();
    bool use_gpu = findall(N, K, A, B, output_cpu.data(), output_gpu);
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

    hipFree(output_gpu);
    hipDeviceSynchronize();

    return 0;
}