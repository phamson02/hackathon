import torch
import torch.nn as nn

def py_solution(input: torch.Tensor, K):
    M, N = input.shape
    rms_norm = nn.RMSNorm([M, N]).cuda()
    input = rms_norm(input)
    input = input.view(M, N // K, K)
    input_abs = input.abs()
    max = input_abs.max(dim=-1, keepdim=True).values
    input = input / max
    input = input.view(M, N)
    return input


# M, N, K
# 1 <= M*N <= 10^10
# 1 <= K <= N
test_cases = [
    (1, 16, 4),
    (16, 16, 4),
    (128, 128, 8),
    (1024, 1024, 16),
    (1024, 1024, 32),
    (1024, 1024, 64),
    (1024, 1024, 128),
    (1024, 1024, 256),
    (1024, 1024, 512),
    (8192, 8192, 16),
    (8192, 8192, 32),
    (8192, 8192, 64),
    (8192, 8192, 128),
    (8192, 8192, 256),
    (8192, 8192, 512),
]


state=2025
def fast_random():
    global state
    state = (state * 48271) % 2147483647
    return state
def random_int(low, high):
    global state
    return low + (fast_random() % (high - low + 1))
def random_float(low, high):
    global state
    return low + (fast_random() % (int)((high - low) * 1000)) / 1000.0


def generate_test_case(M, N, K, input_file, output_file):
    global state
    state = 2025  # Reset state for reproducibility

    input = [random_float(-1.0, 1.0) for _ in range(M * N)]

    with open(input_file, 'w') as f:
        f.write(f"{M} {N} {K}\n")

    output = py_solution(
        torch.tensor(input).view(M, N).cuda(),
        K
    )

    with open(output_file, 'w') as f:
        output = output.flatten().tolist()
        for value in output:
            f.write(f"{value:.9f} ")
        f.write("\n")


if __name__ == "__main__":
    counter = 0
    for M, N, K in test_cases:
        counter += 1
        print(f"Generated test case {counter}: M={M}, N={N}, K={K}")
        input_file = f"data/{str(counter).zfill(3)}.in"
        output_file = f"data/{str(counter).zfill(3)}.ans"
        generate_test_case(M, N, K, input_file, output_file)
        if counter >= 100:
            print("Warning: More than 100 test cases generated, consider reviewing the test cases.")