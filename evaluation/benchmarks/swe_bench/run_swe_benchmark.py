import asyncio
import copy
import json
import os
import tempfile
from typing import Any, Literal

import pandas as pd
import toml
from datasets import load_dataset
from jinja2 import Environment, FileSystemLoader

import openhands.agenthub
from evaluation.benchmarks.swe_bench.binary_patch_utils import (
    remove_binary_diffs,
    remove_binary_files_from_git,
)
from evaluation.benchmarks.swe_bench.resource.mapping import (
    get_instance_resource_factor,
)
from evaluation.benchmarks.swe_bench.resource.swt_bench_constants import (
    MAP_REPO_TO_INSTALL,
    MAP_REPO_TO_TEST_FRAMEWORK_VERBOSE,
    MAP_VERSION_TO_INSTALL,
)
from evaluation.utils.shared import (
    EvalException,
    EvalMetadata,
    EvalOutput,
    BuildOutput,
    assert_and_raise,
    codeact_user_response,
    get_default_sandbox_config_for_eval,
    get_metrics,
    is_fatal_evaluation_error,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
    update_llm_config_for_completions_logging,
)
from evaluation.utils.poisson_scheduler import run_evaluation_poisson
from openhands.controller.state.state import State
from openhands.core.config import (
    AgentConfig,
    OpenHandsConfig,
    get_llm_config_arg,
    get_parser,
)
from openhands.core.config.condenser_config import NoOpCondenserConfig
from openhands.core.config.utils import get_condenser_config_arg
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.critic import AgentFinishedCritic
from openhands.events.action import CmdRunAction, FileReadAction, MessageAction
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    FileReadObservation,
)
from openhands.events.serialization.event import event_from_dict, event_to_dict
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync
from openhands.utils.shutdown_listener import sleep_if_should_continue
from openhands.core.config.llm_config import LLMConfig

USE_HINT_TEXT = os.environ.get('USE_HINT_TEXT', 'false').lower() == 'true'
RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'
ENABLE_LLM_EDITOR = os.environ.get('ENABLE_LLM_EDITOR', 'false').lower() == 'true'
BenchMode = Literal['swe', 'swt', 'swt-ci']

# Global variable to track dataset type
DATASET_TYPE = 'SWE-bench'


def set_dataset_type(dataset_name: str) -> str:
    """Set dataset type based on dataset name."""
    global DATASET_TYPE
    name_lower = dataset_name.lower()

    if 'swe-gym' in name_lower:
        DATASET_TYPE = 'SWE-Gym'
    elif 'swe-bench-live' in name_lower:
        DATASET_TYPE = 'SWE-bench-Live'
    elif 'multimodal' in name_lower:
        DATASET_TYPE = 'Multimodal'
    else:
        DATASET_TYPE = 'SWE-bench'

    logger.info(f'Dataset type set to: {DATASET_TYPE}')


AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response,
}


def _get_swebench_workspace_dir_name(instance: pd.Series) -> str:
    if DATASET_TYPE == 'SWE-bench-Live':
        return instance.instance_id
    else:
        return f'{instance.repo}__{instance.version}'.replace('/', '__')


def get_instruction(instance: pd.Series, metadata: EvalMetadata) -> MessageAction:
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    mode = metadata.details['mode']
    llm_model = metadata.llm_config.model

    # Determine the template file based on mode and LLM
    if mode.startswith('swt'):
        template_name = 'swt.j2'
    elif mode == 'swe':
        if 'claude' in llm_model:
            template_name = 'swe_claude.j2'
        elif 'gemini' in llm_model:
            template_name = 'swe_gemini.j2'
        elif 'gpt-4.1' in llm_model:
            template_name = 'swe_gpt4.j2'
        else:
            template_name = (
                'swe_default.j2'  # Default for 'swe' mode (regular swe-bench)
            )
    else:
        # Fallback or error handling if mode is unexpected
        logger.error(f'Unexpected evaluation mode: {mode}. Falling back to default.')
        template_name = 'swe_default.j2'

    # Set up Jinja2 environment
    # Assuming templates are in 'evaluation/benchmarks/swe_bench/prompts' relative to this script
    prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    # Prepare context for rendering
    context = {
        'instance': instance,
        'workspace_dir_name': workspace_dir_name,
        'metadata': metadata,  # Pass metadata if needed in templates
    }

    # Add specific context for swt-ci mode if needed
    if mode == 'swt-ci':
        context['test_instructions'] = (
            f'The following command can be used to run the tests: `{list(MAP_REPO_TO_TEST_FRAMEWORK_VERBOSE[instance.repo].values())[0]}`. Make sure they fail in the expected way.\n'
        )
    else:
        context['test_instructions'] = ''  # Ensure it's defined for other modes

    # Render the instruction
    instruction = template.render(context)

    if RUN_WITH_BROWSING:
        instruction += (
            '<IMPORTANT!>\nYou SHOULD NEVER attempt to browse the web. </IMPORTANT!>\n'
        )

    if 'image_assets' in instance:
        assets = json.loads(instance['image_assets'])
        assert 'problem_statement' in assets, (
            'problem_statement is required in image_assets'
        )
        image_urls = assets['problem_statement']
        return MessageAction(content=instruction, image_urls=image_urls)
    return MessageAction(content=instruction)


# TODO: migrate all swe-bench docker to ghcr.io/openhands
DEFAULT_DOCKER_IMAGE_PREFIX = os.environ.get(
    'EVAL_DOCKER_IMAGE_PREFIX', 'docker.io/xingyaoww/'
)
logger.info(f'Default docker image prefix: {DEFAULT_DOCKER_IMAGE_PREFIX}')

"""
This script runs the SWE-Bench evaluation benchmark.

In addition to standard evaluation, this script supports running evaluation with
a Poisson time distribution for task launches, which can be useful for:
1. Avoiding overloading resources all at once
2. Simulating more realistic traffic patterns
3. Allowing time for system stabilization between task launches

Usage with Poisson distribution:
    python run_swe_benchmark.py --use-poisson --poisson-rate 2.0 --max-concurrent-tasks 10
    
    - use-poisson: Enable Poisson distribution for task launches
    - poisson-rate: Average number of tasks to launch per minute (λ parameter)
    - max-concurrent-tasks: Maximum number of concurrent tasks (optional)
"""


def get_instance_docker_image(
    instance_id: str,
    swebench_official_image: bool = False,
) -> str:
    if swebench_official_image:
        # Official SWE-Bench image
        # swebench/sweb.eval.x86_64.django_1776_django-11333:v1
        # SWE-bench-Live uses the same naming convention as SWE-Bench
        if DATASET_TYPE == 'SWE-bench-Live':
            docker_image_prefix = 'docker.io/starryzhang/'
        elif DATASET_TYPE == 'SWE-bench':
            docker_image_prefix = 'docker.io/swebench/'
        repo, name = instance_id.split('__')
        image_name = f'{docker_image_prefix.rstrip("/")}/sweb.eval.x86_64.{repo}_1776_{name}:latest'.lower()
        logger.debug(f'Using official SWE-Bench image: {image_name}')
        return image_name
    else:
        # OpenHands version of the image
        docker_image_prefix = DEFAULT_DOCKER_IMAGE_PREFIX
        image_name = 'sweb.eval.x86_64.' + instance_id
        image_name = image_name.replace(
            '__', '_s_'
        )  # to comply with docker image naming convention
        return (docker_image_prefix.rstrip('/') + '/' + image_name).lower()


def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> OpenHandsConfig:
    # We use a different instance image for the each instance of swe-bench eval
    use_swebench_official_image = DATASET_TYPE != 'SWE-Gym'

    base_container_image = get_instance_docker_image(
        instance['instance_id'],
        swebench_official_image=use_swebench_official_image,
    )
    logger.info(
        f'Using instance container image: {base_container_image}. '
        f'Please make sure this image exists. '
        f'Submit an issue on https://github.com/All-Hands-AI/OpenHands if you run into any issues.'
    )

    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = base_container_image
    sandbox_config.enable_auto_lint = True
    sandbox_config.use_host_network = False
    # Add platform to the sandbox config to solve issue 4401
    sandbox_config.platform = 'linux/amd64'
    sandbox_config.remote_runtime_resource_factor = get_instance_resource_factor(
        dataset_name=metadata.dataset,
        instance_id=instance['instance_id'],
    )

    config = OpenHandsConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        max_iterations=metadata.max_iterations,
        runtime=os.environ.get('RUNTIME', 'docker'),
        sandbox=sandbox_config,
        # do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
    )

    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config, metadata.eval_output_dir, instance['instance_id']
        )
    )
    # get 'draft_editor' config if exists
    # config.set_llm_config(get_llm_config_arg('draft_editor'), 'draft_editor')

    agent_config = AgentConfig(
        enable_jupyter=False,
        enable_browsing=RUN_WITH_BROWSING,
        enable_llm_editor=ENABLE_LLM_EDITOR,
        enable_mcp=False,
        condenser=metadata.condenser_config,
        enable_prompt_extensions=False,
    )
    config.set_agent_config(agent_config)
    return config


def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,  # this argument is not required
    metadata: EvalMetadata,
):
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    """
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Initialization Fn')
    logger.info('-' * 30)
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    obs: CmdOutputObservation

    # Set instance id and git configuration
    action = CmdRunAction(
        command=f"""echo 'export SWE_INSTANCE_ID={instance['instance_id']}' >> ~/.bashrc && echo 'export PIP_CACHE_DIR=~/.cache/pip' >> ~/.bashrc && echo "alias git='git --no-pager'" >> ~/.bashrc && git config --global core.pager "" && git config --global diff.binary false"""
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to export SWE_INSTANCE_ID and configure git: {str(obs)}',
    )

    action = CmdRunAction(command="""export USER=$(whoami); echo USER=${USER} """)
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to export USER: {str(obs)}')

    # inject the init script
    script_dir = os.path.dirname(__file__)

    # inject the instance info
    action = CmdRunAction(command='mkdir -p /swe_util/eval_data/instances')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to create /swe_util/eval_data/instances: {str(obs)}',
    )

    swe_instance_json_name = 'swe-bench-instance.json'
    with tempfile.TemporaryDirectory() as temp_dir:
        # Construct the full path for the desired file name within the temporary directory
        temp_file_path = os.path.join(temp_dir, swe_instance_json_name)
        # Write to the file with the desired name within the temporary directory
        with open(temp_file_path, 'w') as f:
            if not isinstance(instance, dict):
                json.dump([instance.to_dict()], f)
            else:
                json.dump([instance], f)

        # Copy the file to the desired location
        runtime.copy_to(temp_file_path, '/swe_util/eval_data/instances/')

        # inject the instance swe entry
        if DATASET_TYPE == 'SWE-bench-Live':
            entry_script_path = 'instance_swe_entry_live.sh'
        else:
            entry_script_path = 'instance_swe_entry.sh'
        runtime.copy_to(
            str(os.path.join(script_dir, f'scripts/setup/{entry_script_path}')),
            '/swe_util/',
        )

    action = CmdRunAction(command='cat ~/.bashrc')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to cat ~/.bashrc: {str(obs)}')

    action = CmdRunAction(command='source ~/.bashrc')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if isinstance(obs, ErrorObservation):
        logger.error(f'Failed to source ~/.bashrc: {str(obs)}')
    assert_and_raise(obs.exit_code == 0, f'Failed to source ~/.bashrc: {str(obs)}')

    action = CmdRunAction(command=f'source /swe_util/{entry_script_path}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to source /swe_util/{entry_script_path}: {str(obs)}',
    )

    action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to cd to /workspace/{workspace_dir_name}: {str(obs)}',
    )

    action = CmdRunAction(command='git reset --hard')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to git reset --hard: {str(obs)}')

    action = CmdRunAction(
        command='for remote_name in $(git remote); do git remote remove "${remote_name}"; done'
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to remove git remotes: {str(obs)}')

    if metadata.details['mode'] == 'swt-ci':
        # set up repo
        setup_commands = []
        if instance['repo'] in MAP_REPO_TO_INSTALL:
            setup_commands.append(MAP_REPO_TO_INSTALL[instance['repo']])

        # Run pre-install set up if provided
        install = MAP_VERSION_TO_INSTALL.get(instance['repo'], {}).get(
            instance['version'], []
        )
        if 'pre_install' in install:
            for pre_install in install['pre_install']:
                setup_commands.append(pre_install)

        if 'install' in install:
            setup_commands.append(install['install'])

        for command in setup_commands:
            action = CmdRunAction(command=command)
            action.set_hard_timeout(600)
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if DATASET_TYPE != 'Multimodal' and DATASET_TYPE != 'SWE-bench-Live':
        # Only for non-multimodal datasets, we need to activate the testbed environment for Python
        # SWE-Bench multimodal datasets and SWE-bench-Live are not using the testbed environment
        action = CmdRunAction(command='which python')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0 and 'testbed' in obs.content,
            f'Expected to find python interpreter from testbed, but got: {str(obs)}',
        )

    logger.info('-' * 30)
    logger.info('END Runtime Initialization Fn')
    logger.info('-' * 30)


def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,  # this argument is not required, but it is used to get the workspace_dir_name
) -> dict[str, Any]:
    """Complete the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    If you need to do something in the sandbox to get the correctness metric after
    the agent has run, modify this function.
    """
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Completion Fn')
    logger.info('-' * 30)
    obs: CmdOutputObservation
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)

    action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if obs.exit_code == -1:
        # The previous command is still running
        # We need to kill previous command
        logger.info('The previous command is still running, trying to kill it...')
        action = CmdRunAction(command='C-c')
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        # Then run the command again
        action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if obs.exit_code == -1:
        # The previous command is still running
        # We need to kill previous command
        logger.info('The previous command is still running, trying to ctrl+z it...')
        action = CmdRunAction(command='C-z')
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        # Then run the command again
        action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to cd to /workspace/{workspace_dir_name}: {str(obs)}',
    )

    action = CmdRunAction(command='git config --global core.pager ""')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to git config --global core.pager "": {str(obs)}',
    )

    # First check for any git repositories in subdirectories
    action = CmdRunAction(command='find . -type d -name .git -not -path "./.git"')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to find git repositories: {str(obs)}',
    )

    git_dirs = [p for p in obs.content.strip().split('\n') if p]
    if git_dirs:
        # Remove all .git directories in subdirectories
        for git_dir in git_dirs:
            action = CmdRunAction(command=f'rm -rf "{git_dir}"')
            action.set_hard_timeout(600)
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})
            assert_and_raise(
                isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
                f'Failed to remove git directory {git_dir}: {str(obs)}',
            )

    # add all files
    action = CmdRunAction(command='git add -A')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to git add -A: {str(obs)}',
    )

    # Remove binary files from git staging
    action = CmdRunAction(command=remove_binary_files_from_git())
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to remove binary files: {str(obs)}',
    )

    n_retries = 0
    git_patch = None
    while n_retries < 5:
        action = CmdRunAction(
            command=f'git diff --no-color --cached {instance["base_commit"]} > patch.diff'
        )
        action.set_hard_timeout(max(300 + 100 * n_retries, 600))
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        n_retries += 1
        if isinstance(obs, CmdOutputObservation):
            if obs.exit_code == 0:
                # Read the patch file
                action = FileReadAction(path='patch.diff')
                action.set_hard_timeout(max(300 + 100 * n_retries, 600))
                logger.info(action, extra={'msg_type': 'ACTION'})
                obs = runtime.run_action(action)
                logger.info(obs, extra={'msg_type': 'OBSERVATION'})
                if isinstance(obs, FileReadObservation):
                    git_patch = obs.content
                    break
                elif isinstance(obs, ErrorObservation):
                    # Fall back to cat "patch.diff" to get the patch
                    assert 'File could not be decoded as utf-8' in obs.content
                    action = CmdRunAction(command='cat patch.diff')
                    action.set_hard_timeout(max(300 + 100 * n_retries, 600))
                    logger.info(action, extra={'msg_type': 'ACTION'})
                    obs = runtime.run_action(action)
                    assert isinstance(obs, CmdOutputObservation) and obs.exit_code == 0
                    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
                    git_patch = obs.content
                    break
                else:
                    assert_and_raise(False, f'Unexpected observation type: {str(obs)}')
            else:
                logger.info('Failed to get git diff, retrying...')
                sleep_if_should_continue(10)
        elif isinstance(obs, ErrorObservation):
            logger.error(f'Error occurred: {obs.content}. Retrying...')
            sleep_if_should_continue(10)
        else:
            assert_and_raise(False, f'Unexpected observation type: {str(obs)}')

    assert_and_raise(git_patch is not None, 'Failed to get git diff (None)')

    # Remove binary diffs from the patch
    git_patch = remove_binary_diffs(git_patch)

    logger.info('-' * 30)
    logger.info('END Runtime Completion Fn')
    logger.info('-' * 30)
    return {'git_patch': git_patch}


def prebuild_runtime_for_instance(
        instance: pd.Series,
        metadata: EvalMetadata,
        reset_logger: bool = True,
):
    from openhands.runtime.builder import DockerRuntimeBuilder
    from openhands.runtime.utils.runtime_build import build_runtime_image
    import docker
    # Prebuild runtime images for the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'build_logs')
        reset_logger_for_multiprocessing(logger, instance.instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.instance_id}.')

    config = get_config(instance, metadata)
    docker_client = docker.from_env()
    builder = DockerRuntimeBuilder(docker_client)
    runtime_container_image = build_runtime_image(
                config.sandbox.base_container_image,
                builder,
                platform=config.sandbox.platform,
                extra_deps=config.sandbox.runtime_extra_deps,
                force_rebuild=config.sandbox.force_rebuild_runtime,
                extra_build_args=config.sandbox.runtime_extra_build_args,
            )
    
    # remove intermediate image
    # final image: ghcr.io/all-hands-ai/runtime:oh_v0.44.0_yfkqmdithaffzkkk_glxm1jpu0685fjbh
    # intermediate image: ghcr.io/all-hands-ai/runtime:oh_v0.44.0_yfkqmdithaffzkkk
    intermediate_image = runtime_container_image.split("_")
    if len(intermediate_image) > 1:
        intermediate_image = "_".join(intermediate_image[:-1])
        logger.info(f'Removing intermediate image: {intermediate_image}')
        try:
            docker_client.images.remove(intermediate_image, force=True)
            logger.info(f'Removed intermediate image: {intermediate_image}')
        except docker.errors.APIError as e:
            logger.error(f'Failed to remove intermediate image {intermediate_image}: {e}')

    return BuildOutput(
        instance_id=instance.instance_id,
        image_name=runtime_container_image
    )


def get_prebulit_runtime_image(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> str:
    lookup_file = os.path.join(
        metadata.eval_output_dir, 'prebuilt_images.jsonl'
    )
    if not os.path.exists(lookup_file):
        logger.warning(
            f'Prebuilt images lookup file {lookup_file} does not exist. '
            'No prebuilt image will be used.'
        )
        return None
    with open(lookup_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if entry['instance_id'] == instance.instance_id:
                return entry['image_name']
    return None


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = False,
    runtime_failure_count: int = 0,
) -> EvalOutput:
    config = get_config(instance, metadata)

    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance.instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.instance_id}.')

    # Increase resource_factor with increasing attempt_id
    if runtime_failure_count > 0:
        config.sandbox.remote_runtime_resource_factor = min(
            config.sandbox.remote_runtime_resource_factor * (2**runtime_failure_count),
            8,
        )
        logger.warning(
            f'This is the {runtime_failure_count + 1}th attempt for instance {instance.instance_id}, setting resource factor to {config.sandbox.remote_runtime_resource_factor}'
        )

    metadata = copy.deepcopy(metadata)
    metadata.details['runtime_failure_count'] = runtime_failure_count
    metadata.details['remote_runtime_resource_factor'] = (
        config.sandbox.remote_runtime_resource_factor
    )

    # change file_storage_path
    config.file_store_path = "/localdisk/minminho/openhands/trajectories/"

    # get prebuilt runtime image if exists
    prebuilt_image = get_prebulit_runtime_image(instance, metadata)
    if prebuilt_image:
        logger.info(
            f'Using prebuilt runtime image {prebuilt_image} for instance {instance.instance_id}'
        )
        config.sandbox.runtime_container_image = prebuilt_image

    # try:
    #     print(f"Config: {config.to_dict()}")
    # except:
    #     print(f"Config:{config}")

    runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)

    try:
        initialize_runtime(runtime, instance, metadata)

        message_action = get_instruction(instance, metadata)
        # print(f"Message Action: {message_action}")

        # Here's how you can run the agent (similar to the `main` function) and get the final task state
        state: State | None = asyncio.run(
            run_controller(
                config=config,
                initial_user_action=message_action,
                runtime=runtime,
                fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN[
                    metadata.agent_class
                ],
            )
        )

        # if fatal error, throw EvalError to trigger re-run
        if is_fatal_evaluation_error(state.last_error):
            raise EvalException('Fatal error detected: ' + state.last_error)

    finally:
        runtime.close()
    # ==========================================

    histories = [event_to_dict(event) for event in state.history]
    output = EvalOutput(
        instance_id=instance.instance_id,
        instruction=message_action.content,
        instance=instance.to_dict(),  # SWE Bench specific
        metadata=metadata,
        history=histories,
        error=state.last_error if state and state.last_error else None,
    )
    return output


def filter_dataset(dataset: pd.DataFrame, filter_column: str, filter_with: str, metadata=None) -> pd.DataFrame:
    if filter_with == 'config_toml':
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.toml')
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = toml.load(file)
                if 'selected_ids' in data:
                    selected_ids = data['selected_ids']
                    logger.info(
                        f'Filtering {len(selected_ids)} tasks from "selected_ids"...'
                    )
                    subset = dataset[dataset[filter_column].isin(selected_ids)]
                    logger.info(f'Retained {subset.shape[0]} tasks after filtering')
                    return subset
                if 'selected_repos' in data:
                    # repos for the swe-bench instances:
                    # ['astropy/astropy', 'django/django', 'matplotlib/matplotlib', 'mwaskom/seaborn', 'pallets/flask', 'psf/requests', 'pydata/xarray', 'pylint-dev/pylint', 'pytest-dev/pytest', 'scikit-learn/scikit-learn', 'sphinx-doc/sphinx', 'sympy/sympy']
                    selected_repos = data['selected_repos']
                    if isinstance(selected_repos, str):
                        selected_repos = [selected_repos]
                    assert isinstance(selected_repos, list)
                    logger.info(
                        f'Filtering {selected_repos} tasks from "selected_repos"...'
                    )
                    subset = dataset[dataset['repo'].isin(selected_repos)]
                    logger.info(f'Retained {subset.shape[0]} tasks after filtering')
                    return subset
    elif filter_with == 'prebuilt_images':
        # Filter based on prebuilt images
        prebuilt_images_file = os.path.join(
            metadata.eval_output_dir, 'prebuilt_images.jsonl'
        )
        if os.path.exists(prebuilt_images_file):
            with open(prebuilt_images_file, 'r') as f:
                prebuilt_images = {json.loads(line)['instance_id'] for line in f}
            logger.info(
                f'Filtering {len(prebuilt_images)} tasks from "prebuilt_images"...'
            )
            return dataset[dataset[filter_column].isin(prebuilt_images)]
        else:
            logger.warning(
                f'Prebuilt images file {prebuilt_images_file} does not exist. No filtering will be applied.'
            )
            logger.warning(
                'If you want to filter tasks based on prebuilt images, please run the prebuild step first.'
            )
            return dataset
    elif filter_with == 'skip_ids':
        skip_ids = os.environ.get('SKIP_IDS', '').split(',')
        if len(skip_ids) > 0:
            logger.info(f'Filtering {len(skip_ids)} tasks from "SKIP_IDS"...')
            return dataset[~dataset[filter_column].isin(skip_ids)]
        return dataset
    else:
        raise ValueError(
            f'Unsupported filter_with value: {filter_with}. Supported values are "config_toml", "prebuilt_images", "skip_ids".'
        )

    return dataset

def get_args():
    parser = get_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='princeton-nlp/SWE-bench',
        help='data set to evaluate on, either full-test or lite-test',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='split to evaluate on',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='swe',
        choices=['swe', 'swt', 'swt-ci'],
        help="mode to run the evaluation, either 'swe', 'swt', or 'swt-ci'",
    )
    parser.add_argument(
        '--use-poisson',
        action='store_true',
        help='Use Poisson time distribution for launching tasks instead of launching all at once',
    )
    parser.add_argument(
        '--poisson-rate',
        type=float,
        default=2.0,
        help='Average number of tasks to launch per minute when using Poisson distribution (default: 2.0)',
    )
    parser.add_argument(
        '--max-concurrent-tasks',
        type=int,
        default=None,
        help='Maximum number of concurrent tasks when using Poisson distribution (default: unlimited)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4.1',
        help='LLM model to use for evaluation'
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default='http://localhost:8000/v1',
        help='Base URL for the LLM endpoint (default: http://localhost:8000/v1)',
    )
    parser.add_argument(
        '--prebuild',
        action='store_true',
        help='Prebuild runtime images for the evaluation',
    )

    parser.add_argument(
        '--data_filter_type',
        type=str,
        default='prebuilt_images',
        choices=['config_toml', 'prebuilt_images', 'skip_ids'],
        help='Type of data filtering to apply. Options are "config_toml", "prebuilt_images", or "skip_ids".',
    )

    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    
    args = get_args()
    # llm_config = None
    # if args.llm_config:
    #     llm_config = get_llm_config_arg(args.llm_config)
    #     llm_config.log_completions = True
    #     # modify_params must be False for evaluation purpose, for reproducibility and accurancy of results
    #     llm_config.modify_params = False

    llm_config = LLMConfig(
        model=args.model,
        temperature=0.3, # arbitrary value
        api_key="empty", # test local llms served by vllm
        base_url=args.base_url,
        log_completions=True,  # Always log completions for evaluation
        modify_params=False,  # Do not modify params for evaluation
    )

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    # Get condenser config from environment variable
    condenser_name = os.environ.get('EVAL_CONDENSER')
    if condenser_name:
        condenser_config = get_condenser_config_arg(condenser_name)
        if condenser_config is None:
            raise ValueError(
                f'Could not find Condenser config: EVAL_CONDENSER={condenser_name}'
            )
    else:
        # If no specific condenser config is provided via env var, default to NoOpCondenser
        condenser_config = NoOpCondenserConfig()
        logger.debug(
            'No Condenser config provided via EVAL_CONDENSER, using NoOpCondenser.'
        )

    details = {'mode': args.mode}
    _agent_cls = openhands.agenthub.Agent.get_cls(args.agent_cls)

    dataset_descrption = (
        args.dataset.replace('/', '__') + '-' + args.split.replace('/', '__')
    )
    metadata = make_metadata(
        llm_config,
        dataset_descrption,
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
        details=details,
        condenser_config=condenser_config,
    )

    # NOTE: It is preferable to load datasets from huggingface datasets and perform post-processing
    # so we don't need to manage file uploading to OpenHands's repo
    dataset = load_dataset(args.dataset, split=args.split)

    # Set the global dataset type based on dataset name
    set_dataset_type(args.dataset)

    swe_bench_tests = filter_dataset(dataset.to_pandas(), 'instance_id', args.data_filter_type, metadata)
    logger.info(
        f'Loaded dataset {args.dataset} with split {args.split}: {len(swe_bench_tests)} tasks'
    )
    if DATASET_TYPE == 'SWE-Gym':
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'split',
                'swegym_verified_instances.json',
            ),
            'r',
        ) as f:
            swegym_verified_instances = json.load(f)
            swe_bench_tests = swe_bench_tests[
                swe_bench_tests['instance_id'].isin(swegym_verified_instances)
            ]
        logger.info(
            f'{len(swe_bench_tests)} tasks left after filtering for SWE-Gym verified instances'
        )


    # prepare_dataset will lookup instances that are already run
    if args.prebuild:
        output_file = os.path.join(metadata.eval_output_dir, 'prebuilt_images.jsonl')
    else:
        output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')

    instances = prepare_dataset(swe_bench_tests, output_file, args.eval_n_limit)

    if len(instances) > 0 and not isinstance(
        instances['PASS_TO_PASS'][instances['PASS_TO_PASS'].index[0]], str
    ):
        for col in ['PASS_TO_PASS', 'FAIL_TO_PASS']:
            instances[col] = instances[col].apply(lambda x: str(x))

    if args.prebuild:
        logger.info('Prebuilding runtime images for the evaluation...')
        run_evaluation(
            instances,
            metadata,
            output_file,
            args.eval_num_workers,
            prebuild_runtime_for_instance,
            timeout_seconds=8 * 60 * 60,  # 8 hour PER instance should be more than enough
            max_retries=2,
        )
        logger.info(f'Prebuilt runtime images saved to {output_file}')

    else:
        if args.use_poisson:
            logger.info(f"Using Poisson time distribution with rate {args.poisson_rate} tasks per minute")
            run_evaluation_poisson(
                instances,
                metadata,
                output_file,
                rate_per_minute=args.poisson_rate,
                process_instance_func=process_instance,
                timeout_seconds=8 * 60 * 60,  # 8 hour PER instance should be more than enough
                max_retries=5,
                max_concurrent_tasks=args.max_concurrent_tasks,
            )
        else:
            logger.info(f"Using standard parallel evaluation with {args.eval_num_workers} workers")
            run_evaluation(
                instances,
                metadata,
                output_file,
                args.eval_num_workers,
                process_instance,
                timeout_seconds=8 * 60 * 60,  # 8 hour PER instance should be more than enough
                max_retries=5,
            )




