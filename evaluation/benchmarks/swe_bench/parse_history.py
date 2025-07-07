import json
import os
from datetime import datetime

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

    history_dict = {}
    for instance in history:
        instance_events = []
        if 'instance_id' not in instance:
            print(f"Skipping entry without instance_id: {instance}")
        instance_id = instance.get('instance_id')
        instance_history = instance.get('history', [])
        if not instance_history:
            print(f"Skipping instance {instance_id} with empty history")
            continue
        for event in instance_history:
            event_dict ={
                'id': event.get('id', 'unknown'),  # Default to 'unknown' if id is not present
                'timestamp': event.get('timestamp'),
                'source': event.get('source', 'unknown'),  # Default to 'unknown' if source is not present
                'action_or_observation': "action" if 'action' in event else "observation",
                'action_or_observation_content': event.get('action', event.get('observation', '')),
            }
            instance_events.append(event_dict)
        history_dict[instance_id] = instance_events
    return history_dict

def process_timestamps(history_dict):
    instance_times = {}
    for instance_id, events in history_dict.items():
        llm_finished_ids = [event["id"] for event in events if event.get('source') == 'agent' and event.get('action_or_observation') == 'action' and event.get('action_or_observation_content') != 'system']
        llm_started_ids = [id-1 for id in llm_finished_ids]
        runtime_finished_ids = [event["id"] for event in events if event.get('source') == 'agent' and event.get('action_or_observation') == 'observation']
        runtime_started_ids = [id-1 for id in runtime_finished_ids]

        llm_finished_timestamps = [event['timestamp'] for event in events if event['id'] in llm_finished_ids]
        llm_started_timestamps = [event['timestamp'] for event in events if event['id'] in llm_started_ids]
        runtime_finished_timestamps = [event['timestamp'] for event in events if event['id'] in runtime_finished_ids]
        runtime_started_timestamps = [event['timestamp'] for event in events if event['id'] in runtime_started_ids]

        print("llm_finished_ids:", llm_finished_ids)
        print("llm_started_ids:", llm_started_ids)
        print("runtime_finished_ids:", runtime_finished_ids)
        print("runtime_started_ids:", runtime_started_ids)

        # timestamps are in str: 2025-07-01T11:11:53.775934
        # convert them so that we can calculate the time difference
        llm_finished_timestamps = [datetime.fromisoformat(ts) for ts in llm_finished_timestamps]
        llm_started_timestamps = [datetime.fromisoformat(ts) for ts in llm_started_timestamps]
        runtime_finished_timestamps = [datetime.fromisoformat(ts) for ts in runtime_finished_timestamps]
        runtime_started_timestamps = [datetime.fromisoformat(ts) for ts in runtime_started_timestamps]

        assert len(llm_finished_timestamps) == len(llm_started_timestamps), "Mismatched LLM timestamps"
        assert len(runtime_finished_timestamps) == len(runtime_started_timestamps), "Mismatched runtime timestamps"
        assert len(llm_finished_timestamps) == len(runtime_finished_timestamps), "Mismatched LLM and runtime timestamps"

        llm_time = []
        runtime_time = []
        
        for i in range(len(llm_finished_timestamps)):
            llm_time.append((llm_finished_timestamps[i] - llm_started_timestamps[i]).total_seconds())
            runtime_time.append((runtime_finished_timestamps[i] - runtime_started_timestamps[i]).total_seconds())
            print(f"Instance {instance_id} step {i+1}: LLM time: {llm_time[-1]} seconds, Runtime time: {runtime_time[-1]} seconds")

        #calculate the total time of the instance
        start_time_stamp = events[0]['timestamp']
        end_time_stamp = events[-1]['timestamp']
        start_time = datetime.fromisoformat(start_time_stamp)
        end_time = datetime.fromisoformat(end_time_stamp)
        total_time = (end_time - start_time).total_seconds()
        print(f"Total time for instance {instance_id}: {total_time} seconds")
        print(f"Agent time for instance {instance_id}: {sum(llm_time)+sum(runtime_time)} seconds")
        instance_times[instance_id] = {
            'total_time': total_time,
            'llm_time': llm_time,
            'runtime_time': runtime_time,
        }
        print("="*50)
    return instance_times

if __name__ == "__main__":
    WORKDIR= os.getenv("WORKDIR", ".")
    EVAL_DIR = os.path.join(WORKDIR, "OpenHands/evaluation/evaluation_outputs/outputs/")
    TEST="princeton-nlp__SWE-bench_Lite-test/CodeActAgent"
    MODEL="Llama-3.3-70B-Instruct"
    N=3
    OPENHANDS_VERSION="v0.44.0"
    postfix = f"{TEST}/{MODEL}_maxiter_{N}_N_{OPENHANDS_VERSION}-no-hint-run_1"

    history_file = os.path.join(EVAL_DIR, f"{postfix}/output.jsonl")
    history = parse_history(history_file)
    # print(len(history))
    instance_times = process_timestamps(history)

    with open(os.path.join(EVAL_DIR, f"{postfix}/instance_times.json"), 'w') as f:
        json.dump(instance_times, f, indent=4)