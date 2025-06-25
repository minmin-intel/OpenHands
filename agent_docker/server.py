import asyncio
import os
import sys
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging
import json
import uuid
from datetime import datetime
import time

# These imports will be done at startup time instead
# to avoid lazy loading issues during request handling
from openhands.core.config import OpenHandsConfig
from openhands.core.config import AgentConfig, SandboxConfig, LLMConfig
from openhands.core.main import create_runtime, run_controller
from openhands.core.setup import create_agent
from openhands.events.action import MessageAction
from openhands.events.serialization.event import event_to_dict
from openhands.utils.async_utils import call_async_from_sync
from openhands.core.config.condenser_config import NoOpCondenserConfig

# Create default agent config
DEFAULT_AGENT_CONFIG = AgentConfig(
    enable_jupyter=False,
    enable_browsing=False,
    enable_llm_editor=False,
    enable_mcp=False,
    enable_prompt_extensions=False,
    condenser=NoOpCondenserConfig(),
)


# Initialize FastAPI app
app = FastAPI(title="OpenHands Agent API")

@app.on_event("startup")
async def startup_event():
    """Log when the application starts."""
    print("====== APPLICATION STARTUP EVENT TRIGGERED ======")

controller_dict = {}

# Dictionary to store ongoing tasks and their results
tasks_store = {}

class UserRequest(BaseModel):
    """Request model for user messages."""
    controller_id: str
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
        print("==== BenchmarkAgentController.__init__ STARTED ====")
        print("Initializing BenchmarkAgentController...")
        self.id = str(uuid.uuid4())
        print(f"Controller ID: {self.id}")
        
        print("About to configure OpenHands")
        print("Configuring agent...")
        self.config = configure_openhands()
        # self.config = config

        print("About to create agent")
        self.agent = create_agent(self.config)
        self.runtime = None
        try:
            print("About to create runtime...")
            print("Create runtime...")
            self.runtime = create_runtime(self.config)

            print("About to connect runtime...")
            print("Connecting runtime...")
            t0 = time.time()
            call_async_from_sync(self.runtime.connect)
            t1 = time.time()
            print(f"** Runtime connection time: {t1 - t0:.2f} seconds")
            print("Runtime connected successfully.")
        except Exception as e:
            print(f"Failed to connect to runtime: {e}")
            print("==== BenchmarkAgentController.__init__ FAILED ====")
            raise
        print("==== BenchmarkAgentController.__init__ COMPLETED SUCCESSFULLY ====")
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
        max_iterations=int(os.environ.get("max_iterations", "10")),
        runtime="docker",
        file_store='local',
        file_store_path=os.environ.get("TRAJECTORY_PATH", '/app/trajectories')
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

    print(f"Sandbox base image: {os.environ.get('sandbox_base_image')}")

    sandbox_config = SandboxConfig(
        use_host_network=False,
        # large enough timeout, since some testcases take very long to run
        timeout=300,
        api_key=os.environ.get('ALLHANDS_API_KEY', None),
        runtime_startup_env_vars={'NO_CHANGE_TIMEOUT_SECONDS': '30'},
        remote_runtime_api_url=os.environ.get('SANDBOX_REMOTE_RUNTIME_API_URL'),
        keep_runtime_alive=False,
        remote_runtime_init_timeout=3600,
        remote_runtime_api_timeout=120,
        remote_runtime_enable_retries=True,
        remote_runtime_class='sysbox',
        enable_auto_lint = True,
        platform = 'linux/amd64',
        base_container_image =os.environ.get("sandbox_base_image")
    )

    config.sandbox = sandbox_config

    print(f"Config: {config}")

    return config


class CreateRequest(BaseModel):
    """Request model for creating a new agent controller."""
    initialize_now: bool = False  # Whether to initialize the controller immediately

@app.post("/create", response_model=CreateResponse)
async def create_agent_controller(request: CreateRequest = None):
    """Endpoint to create a new agent instance."""
    print("==== CREATE ENDPOINT HANDLER STARTED ====")
    try:
        controller_id = str(uuid.uuid4())
        print(f"Generated controller ID: {controller_id}")
        
        # Initialize the controller now if requested
        if request and request.initialize_now:
            print("Initializing controller immediately as requested")
            try:
                controller = BenchmarkAgentController()
                controller.id = controller_id  # Override the auto-generated ID
                controller_dict[controller_id] = controller
                print(f"Controller initialized with ID: {controller_id}")
            except Exception as e:
                print(f"Failed to initialize controller: {e}")
                print(f"==== CONTROLLER INITIALIZATION FAILED ====")
                # Still return the ID even if initialization failed
        else:
            print("Controller will be initialized in the background or on first use")
            # Start background initialization if needed
            # background_tasks = BackgroundTasks()
            # background_tasks.add_task(initialize_controller, controller_id)
        
        print("==== CREATE ENDPOINT HANDLER COMPLETED SUCCESSFULLY ====")
        return CreateResponse(id=controller_id)
    except Exception as e:
        print(f"Error creating agent: {e}")
        print("==== CREATE ENDPOINT HANDLER FAILED ====")
        raise HTTPException(status_code=500, detail=str(e))

async def initialize_controller(controller_id: str):
    """Initialize a controller in the background."""
    print(f"==== INITIALIZING CONTROLLER {controller_id} IN BACKGROUND ====")
    try:
        controller = BenchmarkAgentController()
        print(f"Controller initialized with ID: {controller_id}")
        controller.id = controller_id  # Override the auto-generated ID
        controller_dict[controller_id] = controller
        print(f"==== CONTROLLER {controller_id} INITIALIZED SUCCESSFULLY ====")
    except Exception as e:
        print(f"Failed to initialize controller {controller_id}: {e}")
        print(f"==== CONTROLLER {controller_id} INITIALIZATION FAILED ====")
        # The controller won't be added to controller_dict if initialization fails

@app.post("/run", response_model=TaskStatus)
async def run_agent(request: UserRequest):
    try:
        print(f"==== RUN ENDPOINT HANDLER STARTED FOR CONTROLLER {request.controller_id} ====")
        controller = controller_dict.get(request.controller_id)
        if not controller:
            print(f"Controller with ID {request.controller_id} not found.")
            raise HTTPException(status_code=404, detail="Controller not found or still initializing")
        
        if not hasattr(controller, 'runtime') or controller.runtime is None:
            print(f"Controller with ID {request.controller_id} is not fully initialized.")
            raise HTTPException(status_code=409, detail="Controller is still initializing")
            
        print(f"Running agent with ID: {request.controller_id} and message: {request.message}")
        result = controller.run(request)
        print(f"==== RUN ENDPOINT HANDLER COMPLETED FOR CONTROLLER {request.controller_id} ====")
        return result
    except Exception as e:
        print(f"Error running agent: {e}")
        print(f"==== RUN ENDPOINT HANDLER FAILED FOR CONTROLLER {request.controller_id} ====")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint that also returns default configurations."""
    return {
        "status": "healthy",
    }


def check_docker():
    """Check if Docker is accessible."""
    try:
        import subprocess
        result = subprocess.run(['docker', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Docker is accessible")
            return True
        else:
            print(f"Docker command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error checking Docker: {e}")
        return False


@app.get("/docker-status")
async def docker_status():
    """Check if Docker is accessible."""
    return {"docker_accessible": check_docker()}


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Start the OpenHands Agent Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=int(os.environ.get("PORT", 8000)), help='Port to bind the server to')
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    
    # Force eager loading of all modules before starting the server
    print("Pre-loading all modules to avoid lazy loading issues...")
    # Import any modules that might cause issues if lazy loaded
    from openhands.core.config import OpenHandsConfig, AgentConfig, SandboxConfig, LLMConfig
    from openhands.core.main import create_runtime, run_controller
    from openhands.core.setup import create_agent
    from openhands.events.action import MessageAction
    from openhands.events.serialization.event import event_to_dict
    from openhands.utils.async_utils import call_async_from_sync
    from openhands.core.config.condenser_config import NoOpCondenserConfig
    print("All modules pre-loaded successfully")
    
    uvicorn.run(app, host=args.host, port=args.port)
    # controller = BenchmarkAgentController()
    # print(f"Agent ID: {controller.id}")

