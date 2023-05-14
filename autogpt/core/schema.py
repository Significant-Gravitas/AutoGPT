import enum
from typing import Any, Callable

import pydantic

############################
##### Config Schema ########
############################


class Configuration(pydantic.BaseModel):
    """A configuration class for the agent.
    TODO: Replace types with correct types.
    """

    system: str
    budget_manager: str
    command_registry: str
    credentials: str
    language_model: str
    memory_backend: str
    message_broker: str
    planner: str
    plugin_manager: str
    workspace: str


############################
### LLM Provider Schema ####
############################


class ChatCompletionModels(str, enum.Enum):
    """Available Chat Completion Models"""

    GTP_3_5_TURBO = "gpt-3.5-turbo"
    GTP_3_5_TURBO_0301 = "gpt-3.5-turbo-0301"
    GTP_4 = "gpt-4"
    GTP_4_0314 = "gpt-4-0314"
    GTP_4_32K = "gpt-4-32k"
    GTP_4_32K_0314 = "gpt-4-32k-0314"


class EmbeddingModels(str, enum.Enum):
    """Available Embedding Models"""

    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    TEXT_SEARCH_ADA_DOC_001 = "text-search-ada-doc-001"


class BaseLLMProvider(pydantic.BaseModel):
    """The base class for all LLM providers."""

    num_retries: int = (10,)
    debug_mode: bool = (False,)

    async def create_chat_completion(self, messages: list) -> str:
        """Create a chat completion"""
        raise NotImplementedError


############################
##### Abilities Schema #####
############################
class AbilityResult(pydantic.BaseModel):
    """A result from an ability execution."""

    ok: bool
    message: str


class BaseAbility(pydantic.BaseModel):
    """Base class for abilities the agent can perform."""

    name: str
    description: str
    # abstract concept of cost for agent planning
    # High values make ability less likely to be chosen
    initiative_cost: int
    method: Callable[..., Any]
    signature: str
    enabled: bool

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError


class BaseAbilityRegistry(pydantic.BaseModel):
    """A registry for abilities."""

    AUTO_GPT_ABILITY_IDENTIFIER: str = "auto_gpt_ability"
    abilities: dict[str, BaseAbility] = {}

    def register_ability(self, ability: BaseAbility) -> None:
        """Registers a ability with the ability registry."""
        self.abilities[ability.name] = ability

    def list_abilities(self) -> None:
        """Lists all the abilities in the ability registry."""
        raise NotImplementedError

    def get_ability(self, ability_name: str) -> BaseAbility:
        """Gets a ability from the ability registry."""
        raise NotImplementedError

    def execute_ability(self, ability_name: str, **kwargs) -> AbilityResult:
        """Executes a ability from the ability registry."""
        raise NotImplementedError

    def import_abilities(self, module_name: str) -> None:
        """Imports abilities from a module."""
        raise NotImplementedError


############################
##### Messaging Schema #####
############################


class BaseMessage(pydantic.BaseModel):
    """Base class for a message that can be sent and received on a channel."""

    from_uid: str
    to_uid: str
    timestamp: int


class BaseMessageChannel(pydantic.BaseModel):
    """Base class for a channel that messages can be sent and received on"""

    id: str
    name: str
    host: str
    port: int

    # Channel statistics
    sent_message_count: int = 0
    sent_bytes_count: int = 0
    received_message_count: int = 0
    received_bytes_count: int = 0

    def __str__(self) -> str:
        f"Channel {self.name}:({self.id}) on {self.host}:{self.port}"
        return f"Channel {self.name}:({self.id}) on {self.host}:{self.port}"

    async def get(self) -> None:
        """Gets a message from the channel."""
        raise NotImplementedError

    async def send(self) -> None:
        """Sends a message to the channel."""
        raise NotImplementedError


class BaseMessageBroker(pydantic.BaseModel):
    """Base class for message brokers that holds all the channels an agent can communicate on."""

    channels: list[BaseMessageChannel]

    def list_channels(self) -> None:
        """Lists all the channels."""
        raise NotImplementedError

    def get_channel(self, channel_uid: str) -> BaseMessageChannel:
        """Gets a channel."""
        raise NotImplementedError

    def get_channel_by_name(self, channel_name: str) -> BaseMessageChannel:
        """Gets a channel by name."""
        raise NotImplementedError

    def add_channel(self, channel: BaseMessageChannel) -> None:
        """Adds a channel."""
        raise NotImplementedError

    async def get_from_channel(self, channel_uid: str) -> BaseMessage:
        """Gets a message from a channel."""
        raise NotImplementedError

    async def send_to_channel(self, channel_uid: str, message: BaseMessage) -> None:
        """Sends a message to a channel."""
        raise NotImplementedError


############################
####### Agent Schema #######
############################


class BaseAgent(pydantic.BaseModel):
    """A Base Agent Class"""

    uid: str
    message_broker: BaseMessageBroker

    async def run(self) -> None:
        """Runs the agent"""
        raise NotImplementedError

    async def stop(self) -> None:
        """Stops the agent"""
        raise NotImplementedError
