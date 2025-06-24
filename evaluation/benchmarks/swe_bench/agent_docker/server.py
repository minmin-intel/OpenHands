import asyncio
import os
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging
import json
import uuid
from datetime import datetime
import time

from openhands.controller.state.state import State
from openhands.core.config import OpenHandsConfig, AgentConfig, SandboxConfig, LLMConfig
from openhands.core.main import create_runtime, run_controller
from openhands.core.setup import create_agent
from openhands.events.action import MessageAction
from openhands.events.serialization.event import event_to_dict
from openhands.utils.async_utils import call_async_from_sync
from openhands.core.config.condenser_config import NoOpCondenserConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("openhands-agent-server")


# Create default agent config
DEFAULT_AGENT_CONFIG = AgentConfig(
    enable_jupyter=False,
    enable_browsing=False,
    enable_llm_editor=False,
    enable_mcp=False,
    enable_prompt_extensions=False,
    condenser=NoOpCondenserConfig(),
)

# Create default sandbox config
DEFAULT_SANDBOX_CONFIG = SandboxConfig(
        use_host_network=False,
        # large enough timeout, since some testcases take very long to run
        timeout=300,
        api_key=os.environ.get('ALLHANDS_API_KEY', None),
        runtime_startup_env_vars={'NO_CHANGE_TIMEOUT_SECONDS': '30'},
        remote_runtime_api_url=os.environ.get('SANDBOX_REMOTE_RUNTIME_API_URL'),
        keep_runtime_alive=True,
        remote_runtime_init_timeout=3600,
        remote_runtime_api_timeout=120,
        remote_runtime_enable_retries=True,
        remote_runtime_class='sysbox',
        enable_auto_lint = True,
        platform = 'linux/amd64',
    )

# Initialize FastAPI app
app = FastAPI(title="OpenHands Agent API")

# Dictionary to store ongoing tasks and their results
tasks_store = {}

class UserRequest(BaseModel):
    """Request model for user messages."""
    message: str
    instance_id: Optional[str] = None
    

class TaskStatus(BaseModel):
    """Response model for task status."""
    task_id: str
    status: str
    start_time: str
    completion_time: Optional[str] = None
    message: Optional[str] = None

class CreateResponse(BaseModel):
    """Response model for create endpoint."""
    id: str

class TaskResponse(BaseModel):
    """Response model for task execution."""
    instance_id: str
    instruction: str
    history: List[Dict[str, Any]]
    error: Optional[str] = None
        

def fake_user_response(state, content=None):
    """A simple function to simulate user responses."""
    return MessageAction(content="Continue with the task.")


class BenchmarkAgentController:
    """A simple agent class to encapsulate the OpenHands agent logic."""
    
    def __init__(self):
        self.id = str(uuid.uuid4())
        print("Configuring agent...")
        self.config = configure_openhands()
        # self.config = config

        self.agent = create_agent(self.config)
        try:
            print("Create runtime...")
            self.runtime = create_runtime(self.config)

            print("Connecting runtime...")
            t0 = time.time()
            call_async_from_sync(self.runtime.connect)
            t1 = time.time()
            print(f"** Runtime connection time: {t1 - t0:.2f} seconds")
            logger.info("Runtime connected successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to runtime: {e}")
            raise
        # initialize the runtime
        # TODO

    async def run(self, request):
        """Run the agent with the provided initial user action."""
        print(f"** Running agent with request: {request}")
        message_action = MessageAction(content=request.message)

        t0 = time.time()
        state = await run_controller(
            config=self.config,
            initial_user_action=message_action,
            agent=self.agent,
            runtime=self.runtime,
            fake_user_response_fn=fake_user_response,
        )
        t1 = time.time()
        print(f"** Total controller run time: {t1 - t0:.2f} seconds")
        histories = [event_to_dict(event) for event in state.history]

        # self.runtime.close()
        
        return TaskResponse(
            instance_id=request.instance_id or str(uuid.uuid4()),
            instruction=message_action.content,
            history=histories,
            error=state.last_error if state and state.last_error else "None",
        )


def configure_openhands() -> OpenHandsConfig:
    """Configure OpenHands with the default settings."""
    # Configure OpenHands
    config = OpenHandsConfig(
        run_as_openhands=False,
        max_iterations=os.environ.get("max_iterations", 10),
        runtime="docker",
        file_store='local',
        file_store_path='/localdisk/minminho/openhands/trajectories/' #'/app/storage',
    )
    
    # Use default LLM config and override with user-provided values
    llm_config =LLMConfig(
        model = os.environ.get("LLM_MODEL", "deepseek/deepseek-chat"),
        api_key = os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url= os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        temperature= 0.3,
    )
    config.set_llm_config(llm_config)

    config.set_agent_config(DEFAULT_AGENT_CONFIG)

    sandbox_config = DEFAULT_SANDBOX_CONFIG.copy()
    print(f"Sandbox base image: {os.environ.get('sandbox_base_image')}")
    sandbox_config.base_container_image =os.environ.get("sandbox_base_image")

    config.sandbox = sandbox_config

    print(f"Config: {config}")

    return config

@app.post("/create", response_model=CreateResponse)
async def create_agent():
    """Endpoint to create a new agent instance."""
    try:
        global agent
        agent = BenchmarkAgentController()
        logger.info(f"Agent created with ID: {agent.id}")
        return CreateResponse(id=agent.id)
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process", response_model=TaskStatus)
async def run_agent(request: UserRequest):
    try:
        result = agent.run(request)
        return result
    except Exception as e:
        logger.error(f"Error running agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint that also returns default configurations."""
    return {
        "status": "healthy",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

