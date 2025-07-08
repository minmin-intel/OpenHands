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
    llm_response = {}
    for instance in history:
        instance_events = []
        instance_llm_response = []
        model_response_id_list = []
        if 'instance_id' not in instance:
            print(f"Skipping entry without instance_id: {instance}")
        instance_id = instance.get('instance_id')
        instance_history = instance.get('history', [])
        if not instance_history:
            print(f"Skipping instance {instance_id} with empty history")
            continue
        for event in instance_history:
            assert event.get('message') is not None, f"Event in instance {instance_id} has no message: {event}"
            # print(event["message"])
            event_dict = {
                'id': event.get('id', 'unknown'),  # Default to 'unknown' if id is not present
                'timestamp': event.get('timestamp'),
                'source': event.get('source', 'unknown'),  # Default to 'unknown' if source is not present
                'action_or_observation': "action" if 'action' in event else "observation",
                'action_or_observation_content': event.get('action', event.get('observation', '')),
                'message': event.get('message', ''),
            }
            instance_events.append(event_dict)
            try:
                model_response = event['tool_call_metadata']['model_response']
                if model_response['id'] not in model_response_id_list:
                    instance_llm_response.append({
                        'id': model_response['id'],
                        'created': model_response.get('created', None),
                        'usage': model_response.get('usage', {}),
                    })
                    model_response_id_list.append(model_response['id'])
            except KeyError:
                # If 'tool_call_metadata' or 'model_response' is not present, skip this part
                pass
        history_dict[instance_id] = instance_events
        llm_response[instance_id] = instance_llm_response
    return history_dict, llm_response

def process_timestamps(history_dict, history_file):
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

        # print("llm_finished_ids:", llm_finished_ids)
        # print("llm_started_ids:", llm_started_ids)
        # print("runtime_finished_ids:", runtime_finished_ids)
        # print("runtime_started_ids:", runtime_started_ids)

        # timestamps are in str: 2025-07-01T11:11:53.775934
        # convert them so that we can calculate the time difference
        llm_finished_timestamps = [datetime.fromisoformat(ts) for ts in llm_finished_timestamps]
        llm_started_timestamps = [datetime.fromisoformat(ts) for ts in llm_started_timestamps]
        runtime_finished_timestamps = [datetime.fromisoformat(ts) for ts in runtime_finished_timestamps]
        runtime_started_timestamps = [datetime.fromisoformat(ts) for ts in runtime_started_timestamps]

        assert len(llm_finished_timestamps) == len(llm_started_timestamps), "Mismatched LLM timestamps"
        assert len(runtime_finished_timestamps) == len(runtime_started_timestamps), "Mismatched runtime timestamps"
        # assert len(llm_finished_timestamps) == len(runtime_finished_timestamps), "Mismatched LLM and runtime timestamps"

        llm_time = []
        runtime_time = []
        
        for i in range(len(llm_finished_timestamps)):
            llm_time.append((llm_finished_timestamps[i] - llm_started_timestamps[i]).total_seconds())
            # print(f"Instance {instance_id} step {i+1}: LLM time: {llm_time[-1]} seconds")

        for i in range(len(runtime_finished_timestamps)):
            runtime_time.append((runtime_finished_timestamps[i] - runtime_started_timestamps[i]).total_seconds())
            # print(f"Instance {instance_id} step {i+1}: Runtime time: {runtime_time[-1]} seconds")
            ############ get outlier details ############
            # if runtime_time[-1] >2:
            #     id = runtime_finished_ids[i]
            #     event_start_details = next((event for event in events if event['id'] == id-1), None)
            #     event_end_details = next((event for event in events if event['id'] == id), None)
            #     outliers_file = history_file.replace("output.jsonl", "outliers.jsonl")
            #     with open(outliers_file, 'a') as f:
            #         json.dump({
            #             'instance_id': instance_id,
            #             'event_start': event_start_details,
            #             'event_end': event_end_details
            #         }, f, indent=4)
            #         f.write("\n")
            ############################################

        #calculate the total time of the instance
        start_time_stamp = events[0]['timestamp']
        end_time_stamp = events[-1]['timestamp']
        start_time = datetime.fromisoformat(start_time_stamp)
        end_time = datetime.fromisoformat(end_time_stamp)
        total_time = (end_time - start_time).total_seconds()
        # print(f"Total time for instance {instance_id}: {total_time} seconds")
        # print(f"Agent time for instance {instance_id}: {sum(llm_time)+sum(runtime_time)} seconds")

        # convert timestamps to string format to save in JSON
        llm_started_timestamps = [ts.isoformat() for ts in llm_started_timestamps]
        llm_finished_timestamps = [ts.isoformat() for ts in llm_finished_timestamps]
        runtime_started_timestamps = [ts.isoformat() for ts in runtime_started_timestamps]
        runtime_finished_timestamps = [ts.isoformat() for ts in runtime_finished_timestamps]

        instance_times[instance_id] = {
            'agent_total_time': total_time,
            'llm_time': llm_time,
            'runtime_time': runtime_time,
            'num_llm_calls': len(llm_finished_ids),
            'num_runtime_actions': len(runtime_finished_ids),
            'agent_start_timestamp': start_time_stamp,
            'agent_end_timestamp': end_time_stamp,
            'llm_start_timestamps': llm_started_timestamps,
            'llm_end_timestamps': llm_finished_timestamps,
            'runtime_start_timestamps': runtime_started_timestamps,
            'runtime_end_timestamps': runtime_finished_timestamps
        }
        # print("="*50)
    return instance_times

def get_runtime_startup_timestamps(log_folder, prebuilt_image_file):
    # get all the logs in the log_folder
    log_files = [f for f in os.listdir(log_folder) if f.endswith('.log')]

    # prebuilt_image_file is a jsonl file
    # each line is in form of {"instance_id": "instance_123", "image_name": "ghcr.io/all-hands-ai/runtime:oh_v0.44.0_lhxsocnmcgxny12c_h1icin3k4bvm8gej"}
    with open(prebuilt_image_file, 'r') as f:
        prebuilt_image_lookup = [json.loads(line) for line in f]  

    print(prebuilt_image_lookup)
    # get all the lines that contain "Starting runtime with image:"
    # this is an example:
    # 2025-07-07 17:01:44,319 - INFO - [runtime 75b8c6a9-b76b-420a-b6c6-1b16016215db-c541749a066c53dd] Starting runtime with image: ghcr.io/all-hands-ai/runtime:oh_v0.44.0_lhxsocnmcgxny12c_h1icin3k4bvm8gej
    # get from each line the timestamp, runtime hash, and the image name
    startup_info = {}
    missing_startup = []
    mismatched = []
    for log_file in log_files:
        with open(os.path.join(log_folder, log_file), 'r') as f:
            instance_id = log_file.split(".")[0].replace("instance_", "")
            lines = f.readlines()
            # get the first occurrence of "Starting runtime with image:"          
            startup_line = [line for line in lines if "Starting runtime with image:" in line]
            if not startup_line:
                print(f"No startup line found in {log_file}")
                missing_startup.append(instance_id)
                continue

            parts = startup_line[0].split(" - ")
            if len(parts) > 2:
                timestamp = parts[0].split(",")[0]  # Get the timestamp part before the comma
                runtime_hash = parts[2].split(" ")[1]
                image_name = parts[2].split(" ")[-1]
                print(f"Instance {instance_id} startup info: timestamp={timestamp}, runtime_hash={runtime_hash}, image_name={image_name}")
                # check if image_name mathces any of the prebuilt images
                prebuilt_image = next((img["image_name"] for img in prebuilt_image_lookup if img['instance_id'] == instance_id), None)
                print(f"Prebuilt image for instance {instance_id}: {prebuilt_image}")
                if not prebuilt_image:
                    mismatched.append(instance_id)
                    continue
                startup_info[instance_id] = {
                    'timestamp': timestamp,
                    'runtime_hash': runtime_hash,
                    'image_name': image_name
                }
                # print(startup_info[instance_id])
        
    print("Missing startup timestamps for instances:", missing_startup)
    print("Mismatched instances:", len(mismatched))

    # now piece together all the logs and get all the lines like this:
    # 2025-07-07 17:02:25,694 - INFO - [runtime 412cb379-9a93-422e-b7b3-b259fdba027f-23399499cf53dbde] Runtime is ready.
    runtime_ready_info = {}
    ready_lines = []
    for log_file in log_files:
        with open(os.path.join(log_folder, log_file), 'r') as f:
            lines = f.readlines()
            # get all lines that contain "Runtime is ready."
            ready_lines.extend([line for line in lines if "Runtime is ready." in line])

    for line in ready_lines:
        parts = line.split(" - ")
        if len(parts) > 2:
            timestamp = parts[0].split(",")[0]
            runtime_hash = parts[2].split(" ")[1]
            runtime_ready_info[runtime_hash] = timestamp

    # now match the startup_info with the runtime_ready_info
    startup_time_list = []
    for instance_id, info in startup_info.items():
        runtime_hash = info['runtime_hash']
        if runtime_hash in runtime_ready_info:
            startup_info[instance_id]['runtime_ready_timestamp'] = runtime_ready_info[runtime_hash]
            startup_time = (datetime.fromisoformat(startup_info[instance_id]['runtime_ready_timestamp']) - datetime.fromisoformat(info['timestamp'])).total_seconds()
            startup_time_list.append(startup_time)
        else:
            print(f"Runtime ready timestamp not found for instance {instance_id} with runtime hash {runtime_hash}")
            startup_info[instance_id]['runtime_ready_timestamp'] = None

    print("Missing runtime ready timestamps for instances:", missing_startup)
    return startup_info, startup_time_list

def parse_llm_server_log(log_file):
    pass


def plot_llm_time_stats(instance_times, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    # llm_time vs. llm_start_timestamps
    llm_times = []
    llm_start_times = []
    for instance_id, data in instance_times.items():
        llm_times.extend(data['llm_time'])
        llm_start_times.extend(data['llm_start_timestamps'])
    llm_times = np.array(llm_times)
    # ts is a string in ISO format, convert it to a timestamp
    llm_start_times = [datetime.fromisoformat(ts) for ts in llm_start_times]
    # convert to timestamps
    llm_start_times = np.array([ts.timestamp() for ts in llm_start_times])

    output_file = os.path.join(output_dir, "llm_time_vs_start_time.png")
    plt.figure(figsize=(10, 6))
    plt.scatter(llm_start_times, llm_times, alpha=0.7)
    plt.title('LLM latency vs. LLM request timestamp')
    plt.xlabel('LLM request timestamp')
    plt.ylabel('LLM latency (seconds)')
    plt.grid()
    plt.savefig(output_file)
    plt.close()

    # LLM calls request timestamp for each instance
    # y axis is each instance id
    # x axis is the llm start timestamps
    output_file = os.path.join(output_dir, "llm_start_timestamps.png")
    plt.figure(figsize=(10, 6))
    n = 1
    for instance_id, data in instance_times.items():
        llm_start_times = [datetime.fromisoformat(ts).timestamp() for ts in data['llm_start_timestamps']]
        plt.scatter(llm_start_times, [n] * len(llm_start_times), alpha=0.7)
        n += 1
        # show the instance id on the y axis
    plt.yticks(range(1, n), list(instance_times.keys()), rotation=45)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.title('LLM requests')
    plt.xlabel('LLM request timestamp')
    plt.ylabel('Instance ID')
    plt.grid()
    plt.savefig(output_file)

    # histogram of llm times
    output_file = os.path.join(output_dir, "llm_latency_histogram.png")
    plt.figure(figsize=(10, 6))
    plt.hist(llm_times, bins=20, alpha=0.7, color='blue')
    plt.title('LLM Latency Histogram')
    plt.xlabel('LLM Latency (seconds)')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig(output_file)

def plot_runtime_startup_stats(startup_time_list, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    # Plot the startup times in a histogram
    output_file = os.path.join(output_dir, "runtime_startup_times.png")
    plt.figure(figsize=(10, 6))
    plt.hist(startup_time_list, bins=5, alpha=0.7, color='blue')
    plt.title('Runtime Startup Times')
    plt.xlabel('Startup Time (seconds)')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig(output_file)
    plt.close()



if __name__ == "__main__":
    WORKDIR= os.getenv("WORKDIR", ".")
    EVAL_DIR = os.path.join(WORKDIR, "OpenHands/evaluation/evaluation_outputs/outputs/")
    TEST="princeton-nlp__SWE-bench_Lite-test/CodeActAgent"
    MODEL="Llama-3.3-70B-Instruct"
    N=100
    OPENHANDS_VERSION="v0.44.0"
    postfix = f"{TEST}/{MODEL}_maxiter_{N}_N_{OPENHANDS_VERSION}-no-hint-run_1"

    history_file = os.path.join(EVAL_DIR, f"{postfix}/output.jsonl")
    history, llm_response = parse_history(history_file)
    with open(os.path.join(EVAL_DIR, f"{postfix}/llm_response.json"), 'w') as f:
        json.dump(llm_response, f, indent=4)
    # print(len(history))
    instance_times = process_timestamps(history, history_file)
    with open(os.path.join(EVAL_DIR, f"{postfix}/instance_times.json"), 'w') as f:
        json.dump(instance_times, f, indent=4)

    ##=========== plot LLM stats =====================
    # output_dir = os.path.join(EVAL_DIR, f"{postfix}/plots")
    # os.makedirs(output_dir, exist_ok=True)
    # plot_llm_time_stats(instance_times, output_dir)

    # #=========== runtime related analysis =====================
    # log_folder = os.path.join(EVAL_DIR, f"{postfix}/infer_logs")
    # prebuilt_image_file =f"{WORKDIR}/openhands/prebuilt_images.jsonl"
    # startup_info, startup_time_list = get_runtime_startup_timestamps(log_folder, prebuilt_image_file)
    # with open(os.path.join(EVAL_DIR, f"{postfix}/runtime_startup_info.json"), 'w') as f:
    #     json.dump(startup_info, f, indent=4)

    # output_dir = os.path.join(EVAL_DIR, f"{postfix}/plots")
    # os.makedirs(output_dir, exist_ok=True)
    # plot_runtime_startup_stats(startup_time_list, output_dir)
