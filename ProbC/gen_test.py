
def py_solution(A, Q):
    res = []
    N = len(A)
    for query in Q:
        p, x = query
        if p > 0:
            # bin-search for row p-1
            row = p - 1
            low, high = 0, N - 1
            while low < high:
                mid = (low + high) // 2
                if A[row][mid] < x:
                    low = mid + 1
                else:
                    high = mid - 1
            res.append(low)
        else:
            # bin-search for col -p
            col = -p - 1
            low, high = 0, N - 1
            while low < high:
                mid = (low + high) // 2
                if A[mid][col] < x:
                    low = mid + 1
                else:
                    high = mid - 1
            res.append(low)
    return res


# 2 <= N, K <= 100000
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

    A = [[0] * N for _ in range(N)]
    for j in range(N):
        A[0][j] = j
    for i in range(1, N):
        for j in range(N):
            A[i][j] = A[i-1][j] + 1

    Q = []
    for _ in range(K):
        p = random_int(1, N) * (1 if random_int(0, 1) == 1 else -1)
        if p > 0:
            x = p - 1 + random_int(0, N)
        else:
            x = -p - 1 + random_int(0, N)
        Q.append((p, x))

    with open(input_file, 'w') as f:
        f.write(f"{N} {K}\n")

    output = py_solution(A, Q)

    with open(output_file, 'w') as f:
        for res in output:
            f.write(f"{res} ")
        f.write("\n")


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
