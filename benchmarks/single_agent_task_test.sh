config_file=benchmark_config.toml
session_id=TEST
trajectory_path="/localdisk/minminho/openhands/trajectories"
LLM_MODEL="deepseek/deepseek-chat"

echo "Starting Docker run at $(date)"
start_time=$(date +%s)

docker run -it \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.44-nikolaik \
    -e SANDBOX_USER_ID=$(id -u) \
    -e LOG_ALL_EVENTS=true \
    -e LLM_API_KEY=$DEEPSEEK_API_KEY \
    -e LLM_MODEL=$LLM_MODEL \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /localdisk/minminho/.openhands:/.openhands \
    -v $trajectory_path:/.openhands-state \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app-$(date +%Y%m%d%H%M%S) \
    docker.all-hands.dev/all-hands-ai/openhands:0.44 \
    python -m openhands.core.main -t "write a bash script that prints hi" -n $session_id

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "Docker command completed at $(date)"
echo "Total execution time: $duration seconds"




# -e SANDBOX_VOLUMES=$SANDBOX_VOLUMES \
#     -e LLM_API_KEY=$LLM_API_KEY \
#     -e LLM_MODEL=$LLM_MODEL \

    # --pull=always \
    # --config $config_file