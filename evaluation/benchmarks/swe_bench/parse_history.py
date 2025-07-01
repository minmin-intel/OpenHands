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
        # filter to get events whose source is "agent" AND action is not system
        agent_events = [event for event in events if event.get('source') == 'agent' and event.get('action_or_observation_content') != 'system']
        # sort agent events by id in ascending order
        if not agent_events:
            print(f"No agent events found for instance {instance_id}")
            continue
        agent_events.sort(key=lambda x: x['id'])
        print(f"Agent events:\n{agent_events}")
        # get timestamps
        timestamps = [event['timestamp'] for event in agent_events if 'timestamp' in event]
        if not timestamps:
            print(f"No timestamps found for instance {instance_id}")
            continue
        print(f"Instance {instance_id} has {len(timestamps)//2} steps")
        # timestamps are in str: 2025-07-01T11:11:53.775934
        # convert them so that we can calculate the time difference
        timestamps = [datetime.fromisoformat(ts) for ts in timestamps]
        if len(timestamps) < 2:
            print(f"Not enough timestamps to calculate time difference for instance {instance_id}")
            continue 
        time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]

        # calculate the time for last agent step
        first_agent_event_id = agent_events[0]['id']
        first_agent_event_timestamp = timestamps[0]
        event_before_first_agent_id = first_agent_event_id - 1
        event_before_first_agent = next((event for event in events if event['id'] == event_before_first_agent_id), None)
        if event_before_first_agent:
            event_before_first_agent_timestamp = datetime.fromisoformat(event_before_first_agent['timestamp'])
            time_diffs.insert(0, (first_agent_event_timestamp - event_before_first_agent_timestamp).total_seconds())

        llm_time = []
        runtime_time = []
        for i, time_diff in enumerate(time_diffs):
            if agent_events[i]['action_or_observation'] == 'action':
                llm_time.append(time_diff)
                print(f"Step {i+1} in instance {instance_id}: LLM - {time_diff} seconds")
            else:
                runtime_time.append(time_diff)
                print(f"Step {i+1} in instance {instance_id}: Runtime - {time_diff} seconds")


        #calculate the total time of the instance
        start_time_stamp = events[0]['timestamp']
        end_time_stamp = events[-1]['timestamp']
        start_time = datetime.fromisoformat(start_time_stamp)
        end_time = datetime.fromisoformat(end_time_stamp)
        total_time = (end_time - start_time).total_seconds()
        print(f"Total time for instance {instance_id}: {total_time} seconds")
        print(f"Agent time for instance {instance_id}: {sum(time_diffs)} seconds")
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
    N=6
    OPENHANDS_VERSION="v0.44.0"
    postfix = f"{TEST}/{MODEL}_maxiter_{N}_N_{OPENHANDS_VERSION}-no-hint-run_1"

    history_file = os.path.join(EVAL_DIR, f"{postfix}/output.jsonl")
    history = parse_history(history_file)
    # print(len(history))
    instance_times = process_timestamps(history)

    with open(os.path.join(EVAL_DIR, f"{postfix}/instance_times.json"), 'w') as f:
        json.dump(instance_times, f, indent=4)