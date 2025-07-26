#include <iostream>
#include <chrono>
#include <vector>
#include <cstdint>
#include <iomanip>

#include <hip/hip_runtime.h>

using namespace std;

bool moe(int num_token, int model_dim, int inter_dim, int num_experts, int topk, float* hidden_states_cpu, float* w1_cpu, float* w2_cpu, float* topk_weight_cpu, int* topk_ids_cpu, float* output_cpu, float* hidden_states_gpu, float* w1_gpu, float* w2_gpu, float* topk_weight_gpu, int* topk_ids_gpu, float* output_gpu);

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
    int num_token, model_dim, inter_dim, num_experts, topk;
    cin >> num_token >> model_dim >> inter_dim >> num_experts >> topk;

    float* hidden_states_gpu;
    float* w1_gpu;
    float* w2_gpu;
    float* topk_weight_gpu;
    int* topk_ids_gpu;
    float* output_gpu;
    
    vector<float> hidden_states_cpu(1ll * num_token * model_dim);
    for (size_t i = 0; i < hidden_states_cpu.size(); ++i)
        hidden_states_cpu[i] = random_float(-1.0f, 1.0f) / 10;
    hipMalloc(&hidden_states_gpu, hidden_states_cpu.size() * sizeof(float));
    hipMemcpy(hidden_states_gpu, hidden_states_cpu.data(), hidden_states_cpu.size() * sizeof(float), hipMemcpyHostToDevice);

    vector<float> w1_cpu(1ll * num_experts * inter_dim * 2 * model_dim);
    for (size_t i = 0; i < w1_cpu.size(); ++i)
        w1_cpu[i] = random_float(-1.0f, 1.0f) / 10;
    hipMalloc(&w1_gpu, w1_cpu.size() * sizeof(float));
    hipMemcpy(w1_gpu, w1_cpu.data(), w1_cpu.size() * sizeof(float), hipMemcpyHostToDevice);

    vector<float> w2_cpu(1ll * num_experts * model_dim * inter_dim);
    for (size_t i = 0; i < w2_cpu.size(); ++i)
        w2_cpu[i] = random_float(-1.0f, 1.0f) / 10;
    hipMalloc(&w2_gpu, w2_cpu.size() * sizeof(float));
    hipMemcpy(w2_gpu, w2_cpu.data(), w2_cpu.size() * sizeof(float), hipMemcpyHostToDevice);

    vector<float> topk_weight_cpu(1ll * num_token * topk);
    for (size_t i = 0; i < topk_weight_cpu.size(); ++i)
        topk_weight_cpu[i] = random_float(-1.0f, 1.0f);
    hipMalloc(&topk_weight_gpu, topk_weight_cpu.size() * sizeof(float));
    hipMemcpy(topk_weight_gpu, topk_weight_cpu.data(), topk_weight_cpu.size() * sizeof(float), hipMemcpyHostToDevice);

    vector<int> topk_ids_cpu(1ll * num_token * topk);
    for (int i = 0; i < topk_ids_cpu.size(); ++i)
        topk_ids_cpu[i] = random_int(0, num_experts - 1);
    hipMalloc(&topk_ids_gpu, topk_ids_cpu.size() * sizeof(int));
    hipMemcpy(topk_ids_gpu, topk_ids_cpu.data(), topk_ids_cpu.size() * sizeof(int), hipMemcpyHostToDevice);

    vector<float> output_cpu(1ll * num_token * model_dim);
    hipMalloc(&output_gpu, output_cpu.size() * sizeof(float));

    hipDeviceSynchronize();
    auto start = chrono::high_resolution_clock::now();
    bool use_gpu = moe(num_token, model_dim, inter_dim, num_experts, topk,
        hidden_states_cpu.data(), w1_cpu.data(), w2_cpu.data(),
        topk_weight_cpu.data(), topk_ids_cpu.data(), output_cpu.data(),
        hidden_states_gpu, w1_gpu, w2_gpu, topk_weight_gpu, topk_ids_gpu, output_gpu);
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
        cout << fixed << setprecision(9) << output_cpu[i] << ' ';
    cout << '\n';

    hipFree(hidden_states_gpu);
    hipFree(w1_gpu);
    hipFree(w2_gpu);
    hipFree(topk_weight_gpu);
    hipFree(topk_ids_gpu);
    hipFree(output_gpu);
    hipDeviceSynchronize();

    return 0;
}