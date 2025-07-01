import json
import os

def parse_history(history_file):
    with open(history_file, 'r') as f:
        lines = f.readlines()
    history = []
    for line in lines:
        if line.strip():
            try:
                entry = json.loads(line.strip())
                if isinstance(entry, dict) and 'instance_id' in entry:
                    history.append(entry)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
    timestamp_dict = {}
    source_dict = {}
    for instance in history:
        if 'instance_id' not in instance:
            print(f"Skipping entry without instance_id: {instance}")
        instance_id = instance.get('instance_id')
        instance_history = instance.get('history', [])
        if not instance_history:
            print(f"Skipping instance {instance_id} with empty history")
            continue
        timestamps = [entry['timestamp'] for entry in instance_history if 'timestamp' in entry]
        sources= [entry['source'] for entry in instance_history if 'source' in entry]
        print(f"Timestamps for instance {instance_id}: {timestamps}")
        timestamp_dict[instance_id] = timestamps
        source_dict[instance_id] = sources
    return timestamp_dict, source_dict

def process_timestamps(timestamp_dict, source_dict):
    for instance_id, timestamps in timestamp_dict.items():
        if not timestamps:
            print(f"No timestamps found for instance {instance_id}")
            continue
        print(f"Instance {instance_id} has {len(timestamps)} timestamps")
        # Further processing can be done here if needed
        # timestamps are in str: 2025-07-01T11:11:53.775934
        # convert them so that we can calculate the time difference
        from datetime import datetime
        timestamps = [datetime.fromisoformat(ts) for ts in timestamps]
        if len(timestamps) < 2:
            print(f"Not enough timestamps to calculate time difference for instance {instance_id}")
            continue 
        time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
        print(f"Time differences for instance {instance_id}: {time_diffs}")
        print(f"Total time for instance {instance_id}: {(timestamps[-1]-timestamps[0]).total_seconds()} seconds")
        # filter out timestamps whose source are "agent"
        agent_timestamps = [ts for ts, src in zip(timestamps, source_dict[instance_id]) if src == "agent"]

if __name__ == "__main__":
    WORKDIR= os.getenv("WORKDIR", ".")
    EVAL_DIR = os.path.join(WORKDIR, "OpenHands/evaluation/evaluation_outputs/outputs/")
    TEST="princeton-nlp__SWE-bench_Lite-test/CodeActAgent"
    MODEL="Llama-3.3-70B-Instruct"
    N=6
    OPENHANDS_VERSION="v0.44.0"
    postfix = f"{TEST}/{MODEL}_maxiter_{N}_N_{OPENHANDS_VERSION}-no-hint-run_1"

    history_file = os.path.join(EVAL_DIR, f"{postfix}/output.jsonl")
    history = parse_history(history_file)
    # print(len(history))
    process_timestamps(history)