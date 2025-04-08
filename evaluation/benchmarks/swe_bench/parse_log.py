import json
import os
# parse log file and for each line get the following fields:
# token usages:
# 1. completion_tokens
# 2. prompt_tokens
# 3. cache_read_tokens
# llm_metrics:
# 1. accumulated_cost

def parse_log(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Initialize lists to store the values
    completion_tokens = []
    prompt_tokens = []
    cache_read_tokens = []
    accumulated_cost = []

    # Iterate through each line in the log file
    for line in lines:
        if "token_usages" in line:
            # Extract token usages
            parts = line.split(", ")
            for part in parts:
                part = part.replace('"', "")
                if "completion_tokens" in part:
                    print(part)
                    completion_tokens.append(int(part.split(": ")[1]))
                elif "prompt_tokens" in part:
                    prompt_tokens.append(int(part.split(": ")[1]))
                elif "cache_read_tokens" in part:
                    cache_read_tokens.append(int(part.split(": ")[1]))
                print(f"completion_tokens: {completion_tokens[-1]}, prompt_tokens: {prompt_tokens[-1]}, cache_read_tokens: {cache_read_tokens[-1]}")

        if "llm_metrics" in line:
            # Extract llm_metrics
            parts = line.split(", ")
            for part in parts:
                if "accumulated_cost" in part:
                    accumulated_cost.append(float(part.split(": ")[1]))
                    print(f"accumulated_cost: {accumulated_cost[-1]}")

    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cache_read_tokens": cache_read_tokens,
        "accumulated_cost": accumulated_cost
    }

if __name__ == "__main__":
    # Example usage
    WORKDIR=os.environ.get("WORKDIR")
    file_dir = os.path.join(WORKDIR, "OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/deepseek-chat_maxiter_50_N_v0.31.0-no-hint-run_1/")
    log_file = os.path.join(file_dir, "output.jsonl")
    parsed_data = parse_log(log_file)
    print(json.dumps(parsed_data, indent=4))