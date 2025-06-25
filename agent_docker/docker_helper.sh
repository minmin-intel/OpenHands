#!/bin/bash
# Docker helper script for operations inside the container

# Check if Docker is accessible
docker_check() {
    if docker info > /dev/null 2>&1; then
        echo "Docker is accessible"
        return 0
    else
        echo "Docker is NOT accessible"
        return 1
    fi
}

# Run a Docker command and return the result
docker_run() {
    docker "$@"
}

# Main command execution
case "$1" in
    check)
        docker_check
        ;;
    *)
        docker_run "$@"
        ;;
esac
