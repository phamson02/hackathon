import torch
import torch.nn.functional as F


def py_solution(hidden_states: torch.Tensor,  # [num_token, model_dim]
                w1: torch.Tensor,  # [num_expert, inter_dim*2, model_dim]
                w2: torch.Tensor,  # [num_expert, model_dim, inter_dim]
                topk_weight: torch.Tensor,  # [num_token, topk]
                topk_ids: torch.Tensor,  # [num_token, topk]
                ):
    token_num = hidden_states.shape[0]
    topk = topk_weight.shape[1]
    expert, model_dim, inter_dim = w2.shape
    hidden_states = hidden_states.view(
        token_num, 1, model_dim).repeat(1, topk, 1)
    out = torch.zeros(
        (token_num, topk, model_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    for E_id in range(expert):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
            act_out = F.silu(gate) * up
            out[mask] = act_out @ (w2[E_id].transpose(0, 1))
    return (out * topk_weight.view(token_num, -1, 1)).sum(dim=1)


# num_token, num_expert, model_dim, inter_dim, topk
# 1 <= num_token <= 512
# 1 <= model_dim <= 256
# 1 <= inter_dim <= 2048
# 1 <= topk <= num_expert <= 128
# 1 <= topk <= 8
test_cases = [

    (1, 4, 4, 4, 1),
    (8, 4, 4, 4, 1),
    (64, 4, 4, 4, 1),

    (1, 16, 128, 512, 4),
    (8, 16, 128, 512, 4),
    (64, 16, 128, 512, 4),
    (512, 16, 128, 512, 4),
    (1, 64, 128, 64, 4),
    (8, 64, 128, 64, 4),
    (64, 64, 128, 64, 4),
    (512, 64, 128, 64, 4),

    (1, 16, 128, 1024, 4),
    (8, 16, 128, 1024, 4),
    (64, 16, 128, 1024, 4),
    (512, 16, 128, 1024, 4),
    (1, 64, 128, 128, 4),
    (8, 64, 128, 128, 4),
    (64, 64, 128, 128, 4),
    (512, 64, 128, 128, 4),

    (1, 32, 256, 1024, 8),
    (8, 32, 256, 1024, 8),
    (64, 32, 256, 1024, 8),
    (512, 32, 256, 1024, 8),
    (1, 128, 256, 128, 8),
    (8, 128, 256, 128, 8),
    (64, 128, 256, 128, 8),
    (512, 128, 256, 128, 8),

    (1, 32, 256, 2048, 8),
    (8, 32, 256, 2048, 8),
    (64, 32, 256, 2048, 8),
    (512, 32, 256, 2048, 8),
    (1, 128, 256, 256, 8),
    (8, 128, 256, 256, 8),
    (64, 128, 256, 256, 8),
    (512, 128, 256, 256, 8),
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


def generate_test_case(num_token, num_expert, model_dim, inter_dim, topk, input_file, output_file):
    global state
    state = 2025  # Reset state for reproducibility

    hidden_states = [random_float(-1.0, 1.0) / 10 for _ in range(num_token * model_dim)]
    w1 = [random_float(-1.0, 1.0) / 10 for _ in range(num_expert * inter_dim * 2 * model_dim)]
    w2 = [random_float(-1.0, 1.0) / 10 for _ in range(num_expert * model_dim * inter_dim)]
    topk_weight = [random_float(-1.0, 1.0) for _ in range(num_token * topk)]
    topk_ids = [random_int(0, num_expert - 1) for _ in range(num_token * topk)]

    with open(input_file, 'w') as f:
        f.write(f"{num_token} {num_expert} {model_dim} {inter_dim} {topk}\n")

    output = py_solution(
        torch.tensor(hidden_states).view(num_token, model_dim).cuda(),
        torch.tensor(w1).view(num_expert, inter_dim * 2, model_dim).cuda(),
        torch.tensor(w2).view(num_expert, model_dim, inter_dim).cuda(),
        torch.tensor(topk_weight).view(num_token, topk).cuda(),
        torch.tensor(topk_ids).view(num_token, topk).cuda(),
    )

    with open(output_file, 'w') as f:
        output = output.flatten().tolist()
        for value in output:
            f.write(f"{value:.9f} ")
        f.write("\n")


if __name__ == "__main__":
    counter = 0
    for num_token, num_expert, model_dim, inter_dim, topk in test_cases:
        counter += 1
        print(f"Generating test case {counter}: {num_token} tokens, {num_expert} experts, {model_dim} model_dim, {inter_dim} inter_dim, {topk} topk")
        input_file = f"data/{str(counter).zfill(3)}.in"
        output_file = f"data/{str(counter).zfill(3)}.ans"
        generate_test_case(num_token, num_expert, model_dim, inter_dim, topk, input_file, output_file)
        if counter >= 100:
            print("Warning: More than 100 test cases generated, consider reviewing the test cases.")