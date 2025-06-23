# OpenHands Agent API

This is a FastAPI server that exposes OpenHands agent functionality via a REST API.

## Building the Docker Image

```bash
docker build -t openhands-agent-server .
```

## Running the Server

### Basic Run

```bash
docker run -p 8000:8000 openhands-agent-server
```

### Run with Environment Variables

You can configure the server using environment variables:

```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your-api-key \
  -e DEFAULT_LLM_MODEL=gpt-4 \
  -e OPENAI_API_BASE=https://api.openai.com/v1 \
  -e ENABLE_BROWSING=true \
  -e ENABLE_LLM_EDITOR=true \
  -e BASE_CONTAINER_IMAGE=openhands/agent:latest \
  -e SANDBOX_TIMEOUT_SECONDS=900 \
  -e SANDBOX_MAX_MEMORY_MB=8192 \
  -e ENABLE_AUTO_LINT=true \
  -e USE_HOST_NETWORK=false \
  -e CONTAINER_PLATFORM=linux/amd64 \
  -e REMOTE_RUNTIME_RESOURCE_FACTOR=2.0 \
  openhands-agent-server
```

## API Endpoints

### Start a Task

```
POST /process
```

Example request:
```json
{
  "message": "Create a simple Python function to calculate the factorial of a number",
  "agent_class": "CodeActAgent",
  "max_iterations": 10,
  "llm_config": {
    "model": "gpt-4",
    "api_key": "your-api-key"
  },
  "agent_config": {
    "enable_browsing": true,
    "enable_llm_editor": true
  },
  "sandbox_config": {
    "base_container_image": "openhands/agent:latest",
    "timeout_seconds": 900,
    "max_memory_mb": 8192
  }
}
```

Note: Any values not provided in `llm_config`, `agent_config`, or `sandbox_config` will use the server defaults.

### Check Task Status

```
GET /status/{task_id}
```

### Get Task Result

```
GET /result/{task_id}
```

### Delete Task

```
DELETE /task/{task_id}
```

### Server Configuration

#### Get Current Configuration

```
GET /config
```

Returns the current server configuration for LLM, agent, and sandbox.

#### Update LLM Configuration

```
PUT /config/llm
```

Example request:
```json
{
  "model": "gpt-4-turbo",
  "api_base": "https://api.openai.com/v1",
  "temperature": 0.5
}
```

#### Update Agent Configuration

```
PUT /config/agent
```

Example request:
```json
{
  "enable_browsing": true,
  "enable_llm_editor": true,
  "enable_jupyter": false
}
```

#### Update Sandbox Configuration

```
PUT /config/sandbox
```

Example request:
```json
{
  "base_container_image": "openhands/agent:latest",
  "timeout_seconds": 1200,
  "max_memory_mb": 16384,
  "enable_auto_lint": true
}
```

### Health Check and Default Configurations

```
GET /health
```

This endpoint returns the current server status and the default configurations for LLM, agent, and sandbox.

## Example Using curl

Starting a task:
```bash
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{"message": "Create a simple Python function to calculate the factorial of a number", "agent_class": "CodeActAgent"}'
```

Checking status:
```bash
curl -X GET "http://localhost:8000/status/{task_id}"
```

Getting results:
```bash
curl -X GET "http://localhost:8000/result/{task_id}"
```

Checking server health and default configurations:
```bash
curl -X GET "http://localhost:8000/health"
```

Updating LLM configuration:
```bash
curl -X PUT "http://localhost:8000/config/llm" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4", "temperature": 0.3}'
```
