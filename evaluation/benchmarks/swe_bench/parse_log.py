import json
import os

def parse_log(log_file):
    with open(log_file, 'r') as f:
        data = f.read()

    data = json.loads(data)

    # for k, v in data.items():
    #     print(f"Key: {k}")

    usage = data["response"]["usage"]    

    prompt_tokens = usage["prompt_tokens"]
    completion_tokens = usage["completion_tokens"]
    cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
    new_tokens = prompt_tokens - cached_tokens

    if usage["completion_tokens_details"]:
        reasoning_tokens = usage["completion_tokens_details"]["reasoning_tokens"]
        # print(f"Reasoning tokens: {reasoning_tokens}")
    else:
        reasoning_tokens = 0
    # print(f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Cached tokens: {cached_tokens}, New tokens: {new_tokens}")

    output_tokens = completion_tokens + reasoning_tokens
    cost = data["cost"]
    # print(f"Cost: {cost}")
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cache_read_tokens": cached_tokens,
        "new_tokens": new_tokens,
        "reasoning_tokens": reasoning_tokens,
        "output_tokens": output_tokens,
        "cost": cost
    }

def get_median_and_max(data):
    """
    Calculate the median and maximum values of a list of numbers.
    """
    if not data:
        return None, None
    sorted_data = sorted(data)
    n = len(sorted_data)
    median = int((sorted_data[n // 2] + sorted_data[(n - 1) // 2]) / 2)
    maximum = sorted_data[-1]
    return median, maximum

if __name__ == "__main__":
    # Example usage
    WORKDIR=os.environ.get("WORKDIR")
    file_dir = os.path.join(WORKDIR, "OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/")
    run_name = "deepseek-chat_maxiter_20_N_v0.31.0-no-hint-run_1"
    print(run_name)
    run_folder = f"{run_name}/llm_completions"
    test_cases = os.listdir(os.path.join(file_dir, run_folder))

    prompt_tokens = []
    completion_tokens = []
    cached_tokens = []
    new_tokens = []
    reasoning_tokens = []
    output_tokens = []
    costs = []
    steps = []
    for tc in test_cases:
        log_files = os.listdir(os.path.join(file_dir, run_folder, tc))
        print(f"Test case {tc} takes {len(log_files)} steps")
        steps.append(len(log_files))
        for log in log_files:
            log_file = os.path.join(file_dir, run_folder, tc, log)
            # print(f"Parsing log file: {log_file}")
            parsed_data = parse_log(log_file)
            prompt_tokens.append(parsed_data["prompt_tokens"])
            completion_tokens.append(parsed_data["completion_tokens"])
            cached_tokens.append(parsed_data["cache_read_tokens"])
            new_tokens.append(parsed_data["new_tokens"])
            reasoning_tokens.append(parsed_data["reasoning_tokens"])
            output_tokens.append(parsed_data["output_tokens"])
            costs.append(parsed_data["cost"])

    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Cached tokens: {cached_tokens}")
    print(f"New tokens: {new_tokens}")
    print(f"Reasoning tokens: {reasoning_tokens}")
    print(f"Output tokens: {output_tokens}")
    # print(f"Costs: {costs}")
    print(f"average # steps: {sum(steps)/len(steps)}")
    median_prompt_tokens, max_prompt_tokens = get_median_and_max(prompt_tokens)
    median_completion_tokens, max_completion_tokens = get_median_and_max(completion_tokens)
    median_cached_tokens, max_cached_tokens = get_median_and_max(cached_tokens)
    median_new_tokens, max_new_tokens = get_median_and_max(new_tokens)
    median_reasoning_tokens, max_reasoning_tokens = get_median_and_max(reasoning_tokens)
    median_output_tokens, max_output_tokens = get_median_and_max(output_tokens)
    print(f"Median prompt tokens: {median_prompt_tokens}, Max prompt tokens: {max_prompt_tokens}")
    print(f"Median completion tokens: {median_completion_tokens}, Max completion tokens: {max_completion_tokens}")
    print(f"Median cached tokens: {median_cached_tokens}, Max cached tokens: {max_cached_tokens}")
    print(f"Median new tokens: {median_new_tokens}, Max new tokens: {max_new_tokens}")
    print(f"Median reasoning tokens: {median_reasoning_tokens}, Max reasoning tokens: {max_reasoning_tokens}")
    print(f"Median output tokens: {median_output_tokens}, Max output tokens: {max_output_tokens}")

    total_cost = sum(costs)
    print(f"Total cost: {total_cost}")
            
