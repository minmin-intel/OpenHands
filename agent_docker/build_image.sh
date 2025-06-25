cd $WORKDIR
docker build \
  --no-cache \
  -f OpenHands/agent_docker/Dockerfile \
  --build-arg OPENHANDS_PATH=OpenHands \
  --build-arg http_proxy=${http_proxy} \
  --build-arg https_proxy=${https_proxy} \
  -t openhands-agent .


    