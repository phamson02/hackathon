
def py_solution(K, items):
    dp = [0] * (K + 1)
    dp[0] = 1
    for item in items:
        for j in range(K, item - 1, -1):
            dp[j] += dp[j - item]
            dp[j] %= 1000
    return sum(dp) % 1000


# 2 <= N, K <= 10000
test_cases = [
    (2, 2),
    (10, 5),
    (10, 10),
    (1000, 2500),
    (1000, 5000),
    (1000, 7500),
    (1000, 10000),
    (10000, 1000),
    (10000, 2500),
    (10000, 5000),
    (10000, 7500),
    (10000, 10000),
    (50000, 10000),
    (100000, 100000),
]


state = 2025
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


def generate_test_case(N, K, input_file, output_file):
    global state
    state = 2025  # Reset state for reproducibility

    items = [random_int(1, min(K, 100)) for _ in range(N)]

    with open(input_file, 'w') as f:
        f.write(f"{N} {K}\n")

    output = py_solution(K, items)

    with open(output_file, 'w') as f:
        f.write(str(output) + "\n")


if __name__ == "__main__":
    counter = 0
    for N, K in test_cases:
        counter += 1
        print(f"Generating test case {counter}: N={N}, K={K}")
        input_file = f"data/{str(counter).zfill(3)}.in"
        output_file = f"data/{str(counter).zfill(3)}.ans"
        generate_test_case(N, K, input_file, output_file)
        if counter >= 100:
            print(
                "Warning: More than 100 test cases generated, consider reviewing the test cases.")
