import random

def py_solution(N, K, A, B):

    def MulMod(a, b, mod):
        res = [0] * N
        for i in range(N):
            for j in range(N):
                res[i] = (res[i] + a[(i+j)%N] * b[j]) % mod
        return res

    def PowMod(a, b, mod):
        if b == 1:
            return a
        res = PowMod(a, b // 2, mod)
        res = MulMod(res, res, mod)
        if b % 2 == 1:
            res = MulMod(res, a, mod)
        return res
    
    a = [0] * N
    a[1] = 1
    a[N-1] = 1

    a = PowMod(a, K, 1000000000)

    return a[(B-A+N)%N]


# N, K
# 1 <= N <= 100000
# 1 <= K <= 10^18
test_cases = [
    (3, 3),
    (5, 10),
    (10, 9),
    (1001, 250000001),
    (1001, 50000000001),
    (1001, 7500000000001),
    (1001, 10000000000000001),
    (10001, 100000001),
    (10001, 25000000001),
    (10001, 50000000001),
    (10001, 750000000001),
    (10001, 10000000000000001),
]

random.seed(2025)

def generate_test_case(N, K, input_file, output_file):
    A = random.randint(0, N-1)
    B = random.randint(0, N-1)

    with open(input_file, 'w') as f:
        f.write(f"{N} {K} {A} {B}\n")

    output = py_solution(N, K, A, B)
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
            print("Warning: More than 100 test cases generated, consider reviewing the test cases.")