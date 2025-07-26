import torch
import sys


def validate(input_file, output_file, submission_file):
    try:

        # Load the reference output
        with open(output_file, 'r') as f:
            ref_output = list(map(float, f.readline().strip().split()))
            ref_output = torch.tensor(ref_output, dtype=torch.int32)

        # Load the submission output
        with open(submission_file, 'r') as f:
            elapsed = int(f.readline().strip())
            submission_output = list(map(float, f.readline().strip().split()))
            submission_output = torch.tensor(submission_output, dtype=torch.int32)

        # Validate output
        if (submission_output == ref_output).all():
            valid = True
        else:
            valid = False

    except:
        valid = False
        elapsed = -1

    return valid, elapsed


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    submission_file = sys.argv[3]

    valid, elapsed = validate(input_file, output_file, submission_file)

    if valid:
        sys.exit(42)
    else:
        sys.exit(43)