docker run -it --rm --pull=always \
    -e LOG_ALL_EVENTS=true \
    -e LLM_API_KEY=$DEEPSEEK_API_KEY \
    -e LLM_MODEL=deepseek/deepseek-chat \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /localdisk/minminho/openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.44


# sepcify runtime container base image for the sandbox
## base container image does not work
curl -X POST "http://localhost:3000/api/settings" \
  -H "Content-Type: application/json" \
  -d '{
    "sandbox_base_container_image": "docker.io/swebench/sweb.eval.x86_64.scikit-learn_1776_scikit-learn-10508:latest",
  }'

# runtime container image works
curl -X POST "http://localhost:3000/api/settings" \
  -H "Content-Type: application/json" \
  -d '{
    "sandbox_runtime_container_image": "docker.io/swebench/sweb.eval.x86_64.scikit-learn_1776_scikit-learn-10508:latest"
  }'



# Then create a new conversation
curl -X POST "http://localhost:3000/api/conversations" \
  -H "Content-Type: application/json" \
  -d '{
    "initial_user_msg": "Hello, I need help with my project"
  }'

  # {"status":"ok","conversation_id":"b65407dc1a31499e8ff9c9c70bb6ac1c","message":null,"conversation_status":"STARTING"}

  curl -X GET "http://localhost:3000/api/conversations/b65407dc1a31499e8ff9c9c70bb6ac1c"


curl -X POST "http://localhost:3000/api/conversations/b65407dc1a31499e8ff9c9c70bb6ac1c/start"

  curl -X POST "http://localhost:3000/api/conversations" \
  -H "Content-Type: application/json" \
  -d '{
    "initial_user_msg": "write a bash script that prints hi",
    "repository": null
  }'

  curl -X POST "http://localhost:8000/api/conversations" \
  -H "Content-Type: application/json" \
  -d '{
    "initial_user_msg": "write a bash script that prints hi",
    "sandbox_base_image": "docker.io/swebench/sweb.eval.x86_64.scikit-learn_1776_scikit-learn-10508:latest"
  }'