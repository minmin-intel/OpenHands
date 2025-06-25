
#!/bin/bash
# First, remove any existing container with the same name
docker rm -f openhands-agent-server 2>/dev/null

# Get the Docker group ID on the host system
DOCKER_GID=$(getent group docker | cut -d: -f3)

sandbox_base_image="docker.io/swebench/sweb.eval.x86_64.scikit-learn_1776_scikit-learn-10508:latest"

# Run the container with Docker socket mounted
# Pass the correct Docker GID to ensure permissions work
docker run --name openhands-agent-server \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e DOCKER_GID=$DOCKER_GID \
  -e PORT=8000 \
  -e sandbox_base_image=$sandbox_base_image \
  -p 8000:8000 \
  openhands-agent

# 
# 
# To see logs
# docker logs -f openhands-agent-server

# # To check if Docker is working inside the container
# echo "Checking Docker inside the container..."
# sleep 2  # Give the container a moment to start
# docker exec openhands-agent-server ./docker_helper.sh check