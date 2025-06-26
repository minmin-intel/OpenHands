import uuid
from typing import Any

from openhands.core.logger import openhands_logger as logger
from openhands.events.action.message import MessageAction
from openhands.experiments.experiment_manager import ExperimentManagerImpl
from openhands.integrations.provider import (
    CUSTOM_SECRETS_TYPE_WITH_JSON_SCHEMA,
    PROVIDER_TOKEN_TYPE,
)
from openhands.integrations.service_types import ProviderType
from openhands.server.data_models.agent_loop_info import AgentLoopInfo
from openhands.server.session.conversation_init_data import ConversationInitData
from openhands.server.shared import (
    ConversationStoreImpl,
    SettingsStoreImpl,
    config,
    conversation_manager,
)
from openhands.server.types import LLMAuthenticationError, MissingSettingsError
from openhands.storage.data_models.conversation_metadata import (
    ConversationMetadata,
    ConversationTrigger,
)
from openhands.utils.conversation_summary import get_default_conversation_title
from openhands.storage.data_models.settings import Settings 
'''
class Settings(BaseModel):
    """
    Persisted settings for OpenHands sessions
    """

    language: str | None = None
    agent: str | None = None
    max_iterations: int | None = None
    security_analyzer: str | None = None
    confirmation_mode: bool | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    remote_runtime_resource_factor: int | None = None
    # Planned to be removed from settings
    secrets_store: UserSecrets = Field(default_factory=UserSecrets, frozen=True)
    enable_default_condenser: bool = True
    enable_sound_notifications: bool = False
    enable_proactive_conversation_starters: bool = True
    user_consents_to_analytics: bool | None = None
    sandbox_base_container_image: str | None = None
    sandbox_runtime_container_image: str | None = None
    mcp_config: MCPConfig | None = None
    search_api_key: SecretStr | None = None
    email: str | None = None
    email_verified: bool | None = None

    model_config = {
        'validate_assignment': True,
    }
'''

dummy_settings = Settings(
    # language=None,
    # agent="CodeActAgent",
    llm_model=config.get_llm_config().model,
    llm_api_key=config.get_llm_config().api_key,
    llm_base_url=config.get_llm_config().base_url,
    max_iterations=5,
)

async def create_new_conversation(
    user_id: str | None,
    git_provider_tokens: PROVIDER_TOKEN_TYPE | None,
    custom_secrets: CUSTOM_SECRETS_TYPE_WITH_JSON_SCHEMA | None,
    selected_repository: str | None,
    selected_branch: str | None,
    initial_user_msg: str | None,
    image_urls: list[str] | None,
    replay_json: str | None,
    conversation_instructions: str | None = None,
    conversation_trigger: ConversationTrigger = ConversationTrigger.GUI,
    attach_convo_id: bool = False,
    git_provider: ProviderType | None = None,
    conversation_id: str | None = None,
    sandbox_base_image: str | None = None,
) -> AgentLoopInfo:
    logger.info(
        'Creating conversation',
        extra={
            'signal': 'create_conversation',
            'user_id': user_id,
            'trigger': conversation_trigger.value,
        },
    )
    logger.info('Loading settings')
    # settings_store = await SettingsStoreImpl.get_instance(config, user_id)
    # settings = await settings_store.load()
    settings = dummy_settings  # For testing purposes, we use dummy settings
    logger.info('Settings loaded')
    print(f'===========Settings===========\n{settings}')

    session_init_args: dict[str, Any] = {}
    if settings:
        session_init_args = {**settings.__dict__, **session_init_args}
        # We could use litellm.check_valid_key for a more accurate check,
        # but that would run a tiny inference.
        print(f'===========Session init args===========\n{session_init_args}')
        if (
            not settings.llm_api_key
            or settings.llm_api_key.get_secret_value().isspace()
        ):
            logger.warning(f'Missing api key for model {settings.llm_model}')
            raise LLMAuthenticationError(
                'Error authenticating with the LLM provider. Please check your API key'
            )

    else:
        logger.warning('Settings not present, not starting conversation')
        raise MissingSettingsError('Settings not found')

    session_init_args['git_provider_tokens'] = git_provider_tokens
    session_init_args['selected_repository'] = selected_repository
    session_init_args['custom_secrets'] = custom_secrets
    session_init_args['selected_branch'] = selected_branch
    session_init_args['git_provider'] = git_provider
    session_init_args['conversation_instructions'] = conversation_instructions
    session_init_args['sandbox_base_container_image'] = sandbox_base_image
    conversation_init_data = ConversationInitData(**session_init_args)
    print(f'===========Conversation init data===========\n{conversation_init_data}')

    logger.info('Loading conversation store')
    conversation_store = await ConversationStoreImpl.get_instance(config, user_id)
    logger.info('ServerConversation store loaded')

    # For nested runtimes, we allow a single conversation id, passed in on container creation
    if conversation_id is None:
        conversation_id = uuid.uuid4().hex

    if not await conversation_store.exists(conversation_id):
        logger.info(
            f'New conversation ID: {conversation_id}',
            extra={'user_id': user_id, 'session_id': conversation_id},
        )

        conversation_init_data = ExperimentManagerImpl.run_conversation_variant_test(
            user_id, conversation_id, conversation_init_data
        )
        conversation_title = get_default_conversation_title(conversation_id)

        logger.info(f'Saving metadata for conversation {conversation_id}')
        await conversation_store.save_metadata(
            ConversationMetadata(
                trigger=conversation_trigger,
                conversation_id=conversation_id,
                title=conversation_title,
                user_id=user_id,
                selected_repository=selected_repository,
                selected_branch=selected_branch,
                git_provider=git_provider,
                llm_model=conversation_init_data.llm_model,
            ) #TODO: check if we need to add sandbox_base_image here
        )

    logger.info(
        f'Starting agent loop for conversation {conversation_id}',
        extra={'user_id': user_id, 'session_id': conversation_id},
    )
    initial_message_action = None
    if initial_user_msg or image_urls:
        initial_message_action = MessageAction(
            content=initial_user_msg or '',
            image_urls=image_urls or [],
        )

    if attach_convo_id:
        logger.warning('Attaching convo ID is deprecated, skipping process')

    agent_loop_info = await conversation_manager.maybe_start_agent_loop(
        conversation_id,
        conversation_init_data,
        user_id,
        initial_user_msg=initial_message_action,
        replay_json=replay_json,
    )
    logger.info(f'Finished initializing conversation {agent_loop_info.conversation_id}')
    return agent_loop_info
