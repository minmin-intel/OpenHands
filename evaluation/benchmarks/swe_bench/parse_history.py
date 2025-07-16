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
    llm_latency_outliers = []
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
                    # get the time stamps of this event and the previous event
                    previous_event = next((e for e in instance_history if e['id'] == event['id'] - 1), None)
                    if previous_event:
                        previous_timestamp = previous_event.get('timestamp', None)
                        current_timestamp = event.get('timestamp', None)
                        if previous_timestamp and current_timestamp:
                            time_diff = (datetime.fromisoformat(current_timestamp) - datetime.fromisoformat(previous_timestamp)).total_seconds()
                    else:
                        model_response['latency'] = None
                    instance_llm_response.append({
                        'id': model_response['id'],
                        'created': model_response.get('created', None),
                        'usage': model_response.get('usage', {}),
                        'event_id': event.get('id', 'unknown'),  # Link back to the event id
                        'latency': time_diff
                    })
                    model_response_id_list.append(model_response['id'])
                    if time_diff > 2000:
                        llm_latency_outliers.append({
                            'instance_id': instance_id,
                            'event_id': event.get('id', 'unknown'),
                            'llm_latency': time_diff,
                            'model_response': model_response,
                            'previous_event_message': previous_event.get('message', ''),
                            'current_event_message': event.get('message', '')
                        })
            except KeyError:
                # If 'tool_call_metadata' or 'model_response' is not present, skip this part
                pass
        history_dict[instance_id] = instance_events
        llm_response[instance_id] = instance_llm_response
    return history_dict, llm_response, llm_latency_outliers

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

        if len(llm_finished_timestamps) != len(llm_started_timestamps):
            print(f"Warning: Mismatched LLM finished and started timestamps for instance {instance_id}.")
            continue
        if len(runtime_finished_timestamps) != len(runtime_started_timestamps):
            print(f"Warning: Mismatched runtime finished and started timestamps for instance {instance_id}.")
            continue

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

    # print(prebuilt_image_lookup)
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
                # print(f"Instance {instance_id} startup info: timestamp={timestamp}, runtime_hash={runtime_hash}, image_name={image_name}")
                # check if image_name mathces any of the prebuilt images
                prebuilt_image = next((img["image_name"] for img in prebuilt_image_lookup if img['instance_id'] == instance_id), None)
                # print(f"Prebuilt image for instance {instance_id}: {prebuilt_image}")
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
    print("Number of Mismatched instances:", len(mismatched))

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

    return startup_info, startup_time_list

def parse_llm_server_log(log_file):
    pass

def calculate_stats_llm_usage(llm_response, output_dir=None, model_name=None):
    import numpy as np
    llm_usage_stats = {}
    for instance_id, responses in llm_response.items():
        completion_tokens = []
        prompt_tokens = []
        total_tokens = []
        new_tokens = []
        for response in responses:
            usage = response.get('usage', {})
            completion_tokens.append(usage.get('completion_tokens', 0))
            prompt_tokens.append(usage.get('prompt_tokens', 0))
            total_tokens.append(usage.get('total_tokens', 0))

        for i in range(1, len(completion_tokens)):
            new_tokens.append(prompt_tokens[i] - total_tokens[i - 1])

        llm_usage_stats[instance_id] = {
            'completion_tokens': completion_tokens,
            'prompt_tokens': prompt_tokens,
            'total_tokens': total_tokens,
            'first_tokens': prompt_tokens[0] if prompt_tokens else 0,
            'new_tokens': new_tokens
        }
    # calculate median and max of all instances
    # for key in ['completion_tokens', 'prompt_tokens', 'total_tokens', 'new_tokens']:
    #     values = [stats[key] for stats in llm_usage_stats.values()]
    #     llm_usage_stats['median_' + key] = np.median(values) if values else 0
    #     llm_usage_stats['max_' + key] = np.max(values) if values else 0
    
    # combine all the tokens into a single list for overall stats
    all_completion_tokens = []
    all_prompt_tokens = []
    all_total_tokens = []
    all_new_tokens = []
    all_first_tokens = []
    for stats in llm_usage_stats.values():
        all_completion_tokens.extend(stats['completion_tokens'])
        all_prompt_tokens.extend(stats['prompt_tokens'])
        all_total_tokens.extend(stats['total_tokens'])
        all_new_tokens.extend(stats['new_tokens'])
        all_first_tokens.append(stats['first_tokens'])

    median_completion_tokens = np.median(all_completion_tokens) if all_completion_tokens else 0
    median_prompt_tokens = np.median(all_prompt_tokens) if all_prompt_tokens else 0
    median_total_tokens = np.median(all_total_tokens) if all_total_tokens else 0
    median_new_tokens = np.median(all_new_tokens) if all_new_tokens else 0
    median_first_tokens = np.median(all_first_tokens) if all_first_tokens else 0

    max_completion_tokens = np.max(all_completion_tokens) if all_completion_tokens else 0
    max_prompt_tokens = np.max(all_prompt_tokens) if all_prompt_tokens else 0
    max_total_tokens = np.max(all_total_tokens) if all_total_tokens else 0
    max_new_tokens = np.max(all_new_tokens) if all_new_tokens else 0
    max_first_tokens = np.max(all_first_tokens) if all_first_tokens else 0


    print("LLM Usage Stats:")
    print(f"Median Completion Tokens: {median_completion_tokens}")
    print(f"Median Prompt Tokens: {median_prompt_tokens}")
    print(f"Median Total Tokens: {median_total_tokens}")
    print(f"Median First Tokens: {median_first_tokens}")
    print(f"Median New Tokens: {median_new_tokens}")

    print(f"Mean Completion Tokens: {np.mean(all_completion_tokens) if all_completion_tokens else 0}")
    print(f"Mean Prompt Tokens: {np.mean(all_prompt_tokens) if all_prompt_tokens else 0}")
    print(f"Mean Total Tokens: {np.mean(all_total_tokens) if all_total_tokens else 0}")
    print(f"Mean First Tokens: {np.mean(all_first_tokens) if all_first_tokens else 0}")     
    print(f"Mean New Tokens: {np.mean(all_new_tokens) if all_new_tokens else 0}")

    print(f"Min Completion Tokens: {min(all_completion_tokens) if all_completion_tokens else 0}")
    print(f"Min Prompt Tokens: {min(all_prompt_tokens) if all_prompt_tokens else 0}")
    print(f"Min Total Tokens: {min(all_total_tokens) if all_total_tokens else 0}")
    print(f"Min First Tokens: {min(all_first_tokens) if all_first_tokens else 0}")
    print(f"Min New Tokens: {min(all_new_tokens) if all_new_tokens else 0}")

    print(f"Max Completion Tokens: {max_completion_tokens}")
    print(f"Max Prompt Tokens: {max_prompt_tokens}")
    print(f"Max Total Tokens: {max_total_tokens}")
    print(f"Max First Tokens: {max_first_tokens}")
    print(f"Max New Tokens: {max_new_tokens}")

    print(f"STD Completion Tokens: {np.std(all_completion_tokens) if all_completion_tokens else 0}")
    print(f"STD Prompt Tokens: {np.std(all_prompt_tokens) if all_prompt_tokens else 0}")
    print(f"STD Total Tokens: {np.std(all_total_tokens) if all_total_tokens else 0}")
    print(f"STD First Tokens: {np.std(all_first_tokens) if all_first_tokens else 0}")
    print(f"STD New Tokens: {np.std(all_new_tokens) if all_new_tokens else 0}")

    # plot histograms for each token type
    import matplotlib.pyplot as plt
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        for key, values in zip(['completion_tokens', 'prompt_tokens'], [all_completion_tokens, all_prompt_tokens]):
            plt.figure(figsize=(10, 6))
            plt.hist(values, bins=20, alpha=0.7, color='blue')
            plt.title(f'{model_name}\nLLM {key.replace("_", " ").title()} Histogram')
            plt.xlabel(key.replace("_", " ").title())
            plt.ylabel('Frequency')
            plt.grid()
            output_file = os.path.join(output_dir, f'llm_{key}_histogram.png')
            plt.savefig(output_file)
            plt.close()
    
    # plot latency vs. completion tokens and prompt tokens and total tokens
    if output_dir:
        for key in ['completion_tokens', 'prompt_tokens', 'total_tokens']:
            plt.figure(figsize=(10, 6))
            for instance_id, responses in llm_response.items():
                latencies = [response.get('latency', 0) for response in responses]
                tokens = [response['usage'].get(key, 0) for response in responses]
                plt.scatter(tokens, latencies, alpha=0.5)
            plt.title(f'{model_name}\nLLM Latency vs {key.replace("_", " ").title()}')
            plt.xlabel(key.replace("_", " ").title())
            plt.ylabel('LLM Latency (seconds)')
            output_file = os.path.join(output_dir, f'llm_latency_vs_{key}.png')
            plt.savefig(output_file)
            plt.close()


    return llm_usage_stats

def get_history_stats(history_dict):
    user_message_lengths = []
    system_message_lengths = []
    for instance_id, events in history_dict.items():
        for event in events:
            if event.get('source') == 'agent' and event.get('action_or_observation_content') == 'system':
                system_message = event.get('message', '')
                system_message_lengths.append(len(system_message))
            if event.get('source') == 'user' and event.get('action_or_observation_content') == 'message':
                user_message = event.get('message', '')
                user_message_lengths.append(len(user_message))
        # print(f"Instance {instance_id} - System Message: {system_message}, User Message: {user_message}")
    
    # Calculate statistics
    total_len = [sys_msg_len + user_msg_len for sys_msg_len, user_msg_len in zip(system_message_lengths, user_message_lengths)]
    avg_total_len = sum(total_len) / len(total_len) if total_len else 0
    avg_system_len = sum(system_message_lengths) / len(system_message_lengths) if system_message_lengths else 0
    avg_user_len = sum(user_message_lengths) / len(user_message_lengths) if user_message_lengths else 0
    print(f"Average System Message Length: {avg_system_len}")
    print(f"Average User Message Length: {avg_user_len}")
    print(f"Average Total Message Length: {avg_total_len}")
    avg_sys_percentage = (avg_system_len/ avg_total_len) * 100 if avg_total_len > 0 else 0
    avg_user_percentage = (avg_user_len / avg_total_len) * 100 if avg_user_len > 0 else 0
    print(f"Average System Message Percentage: {avg_sys_percentage:.2f}%")
    print(f"Average User Message Percentage: {avg_user_percentage:.2f}%")
                

def plot_llm_time_stats(instance_times, output_dir, model_name=None):
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
    plt.title(f'{model_name}\nLLM latency vs. LLM request timestamp')
    plt.xlabel('LLM request timestamp')
    plt.ylabel('LLM latency (seconds)')
    plt.grid()
    plt.savefig(output_file)
    plt.close()

    # LLM calls request timestamp for each instance
    # y axis is each instance id
    # x axis is the llm start timestamps
    output_file = os.path.join(output_dir, "llm_start_timestamps.png")
    plt.figure(figsize=(10, 8))
    n = 1
    for instance_id, data in instance_times.items():
        llm_start_times = [datetime.fromisoformat(ts).timestamp() for ts in data['llm_start_timestamps']]
        plt.scatter(llm_start_times, [n] * len(llm_start_times), alpha=0.7)
        n += 1
        # show the instance id on the y axis
    plt.yticks(range(1, n), list(instance_times.keys()), rotation=45)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.title(f'{model_name}\nLLM requests')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add more space at the top for the title
    plt.xlabel('LLM request timestamp')
    plt.ylabel('Instance ID')
    plt.grid()
    plt.savefig(output_file)

    # histogram of llm times
    output_file = os.path.join(output_dir, "llm_latency_histogram.png")
    plt.figure(figsize=(10, 6))
    plt.hist(llm_times, bins=20, alpha=0.7, color='blue')
    plt.title(f'{model_name}\nLLM Latency Histogram')
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

def calculate_metrics(instance_times, startup_time_list, llm_usage_stats):
    import numpy as np
    # median of startup_time_list
    startup_time_median = np.median(startup_time_list)
    print(f"Median startup time: {startup_time_median:.2f} seconds")

    # median of llm times
    llm_times = []
    for instance_id, data in instance_times.items():
        llm_times.extend(data['llm_time'])
    llm_time_median = np.median(llm_times)
    print(f"Median LLM latency: {llm_time_median:.2f} seconds")

    # median number of LLM calls
    num_llm_calls = [data['num_llm_calls'] for data in instance_times.values()]
    num_llm_calls_median = np.median(num_llm_calls)
    print(f"Median number of LLM calls: {num_llm_calls_median}")

    # throughput
    total_instances = len(instance_times)
    median_e2e_latency = startup_time_median + llm_time_median * num_llm_calls_median
    print(f"Total instances: {total_instances}")
    print(f"Median end-to-end latency: {median_e2e_latency:.2f} seconds")
    throughput = total_instances / median_e2e_latency if median_e2e_latency > 0 else 0
    print(f"Throughput: {throughput:.3f} instances/second")

    # get the earliest and latest timestamps
    all_start_timestamps = []
    all_end_timestamps = []
    for instance_id, data in instance_times.items():
        all_start_timestamps.append(datetime.fromisoformat(data['agent_start_timestamp']))
        all_end_timestamps.append(datetime.fromisoformat(data['agent_end_timestamp']))
    earliest_start = min(all_start_timestamps)
    latest_end = max(all_end_timestamps)
    total_time = (latest_end - earliest_start).total_seconds()
    print(f"Total time: {total_time:.2f} seconds")

    # the total number of llm calls
    total_llm_calls = sum(num_llm_calls)
    print(f"Total number of LLM calls: {total_llm_calls}")
    llm_reqs_per_second = total_llm_calls / total_time if total_time > 0 else 0
    print(f"LLM requests per second: {llm_reqs_per_second:.3f}")

    # total number of prompt tokens
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for instance_id, stats in llm_usage_stats.items():
        total_prompt_tokens += sum(stats['prompt_tokens'])
        total_completion_tokens += sum(stats['completion_tokens'])
    output_tokens_per_second = total_completion_tokens / total_time if total_time > 0 else 0
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print(f"Total completion tokens: {total_completion_tokens}")
    print(f"Output tokens per second: {output_tokens_per_second:.3f}")
    print(f"Total tokens per second: {(total_prompt_tokens + total_completion_tokens) / total_time if total_time > 0 else 0:.3f}")
    


def plot_llm_latency_outliers(llm_latency_outliers, output_dir, model_name=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    # plot latency vs. completion tokens and prompt tokens
    for key in ['completion_tokens', 'prompt_tokens', 'total_tokens']:
        plt.subplot(1, 3, ['completion_tokens', 'prompt_tokens', 'total_tokens'].index(key) + 1)
        plt.title(f'LLM Latency vs. {key}')
        plt.xlabel(key)
        plt.ylabel('LLM Latency (seconds)')
        for outlier in llm_latency_outliers:
            latency = outlier['llm_latency']
            plt.scatter(outlier["model_response"]["usage"][key], latency, alpha=0.5)
    plt.tight_layout()
    plt.suptitle(f'LLM Latency Outliers - {model_name}')
    plt.subplots_adjust(top=0.85)  # Adjust the top margin to make space
    plt.savefig(os.path.join(output_dir, "llm_latency_outliers.png"))
    plt.close()



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Parse SWE-bench history and calculate metrics.")
    parser.add_argument('--dataset', type=str, required=True, default='SWE-bench_Lite', help='name of dataset')
    parser.add_argument('--split', type=str, required=True, default='test', help='Data split (e.g. test, dev)')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum number of iterations for the agent')
    parser.add_argument('--model', type=str, help='Model name to use for the agent')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    WORKDIR= os.getenv("WORKDIR", ".")
    EVAL_DIR = os.path.join(WORKDIR, "OpenHands/evaluation/evaluation_outputs/outputs/")
    TEST=f"princeton-nlp__{args.dataset}-{args.split}/CodeActAgent"
    MODEL=args.model.split("/")[-1]  # get the last part of the model name
    N=args.max_iter
    OPENHANDS_VERSION="v0.44.0"
    postfix = f"{TEST}/{MODEL}_maxiter_{N}_N_{OPENHANDS_VERSION}-no-hint-run_1"

    print("===============Parsing agent history logs=====================")
    history_file = os.path.join(EVAL_DIR, f"{postfix}/output.jsonl")
    history, llm_response, llm_latency_outliers = parse_history(history_file)
    with open(os.path.join(EVAL_DIR, f"{postfix}/llm_response.json"), 'w') as f:
        json.dump(llm_response, f, indent=4)
    with open(os.path.join(EVAL_DIR, f"{postfix}/llm_latency_outliers.json"), 'w') as f:
        json.dump(llm_latency_outliers, f, indent=4)

    print(f"Parsed {len(history)} instances from: {history_file}")
    print('================Parsing timestamps from history====================')
    instance_times = process_timestamps(history, history_file)
    with open(os.path.join(EVAL_DIR, f"{postfix}/instance_times.json"), 'w') as f:
        json.dump(instance_times, f, indent=4)
    print(f"Parsed timestamps for {len(instance_times)} instances.")
    print(f"Saving instance times to: {os.path.join(EVAL_DIR, f'{postfix}/instance_times.json')}")

    ##=========== plot LLM stats =====================
    print("================Plotting LLM latency stats======================")
    output_dir = os.path.join(EVAL_DIR, f"{postfix}/plots")
    os.makedirs(output_dir, exist_ok=True)
    plot_llm_time_stats(instance_times, output_dir, model_name=MODEL)

    plot_llm_latency_outliers(llm_latency_outliers, output_dir, model_name=MODEL)

    # #=========== runtime related analysis =====================
    print("================Parsing runtime containers startup timestamps========================")
    log_folder = os.path.join(EVAL_DIR, f"{postfix}/infer_logs")
    prebuilt_image_file =f"{WORKDIR}/openhands/prebuilt_images.jsonl"
    startup_info, startup_time_list = get_runtime_startup_timestamps(log_folder, prebuilt_image_file)
    with open(os.path.join(EVAL_DIR, f"{postfix}/runtime_startup_info.json"), 'w') as f:
        json.dump(startup_info, f, indent=4)
    print(f"Parsed startup info for {len(startup_info)} instances.")
    print(f"Saving startup info to: {os.path.join(EVAL_DIR, f'{postfix}/runtime_startup_info.json')}")

    ## ================plot startup times histogram =========================
    print("================Plotting histogram for runtime startup times=====================")
    plot_runtime_startup_stats(startup_time_list, output_dir)

    ## ================ get LLM usage stats =========================
    print("================Calculating LLM usage stats=======================")
    llm_usage_stats = calculate_stats_llm_usage(llm_response, output_dir=output_dir, model_name=MODEL)
    with open(os.path.join(EVAL_DIR, f"{postfix}/llm_usage_stats.json"), 'w') as f:
        json.dump(llm_usage_stats, f, indent=4)
    print(f"LLM usage stats saved to: {os.path.join(EVAL_DIR, f'{postfix}/llm_usage_stats.json')}")

    ## ================ calculate metrics =========================
    print("==================Calculating metrics========================")
    calculate_metrics(instance_times, startup_time_list, llm_usage_stats)


    # ## ================ get history stats =========================
    # print("Calculating history stats...")
    # get_history_stats(history)
