import enum
import numpy as np
from typing import Any, Callable, List, Optional, Dict, Literal, Sequence, MutableSet
import pydantic

############################
##### Config Schema ########
############################


class Configuration(pydantic.BaseModel):
    """A configuration class for the agent.
    TODO: Replace types with correct types.
    """

    ability_registry: str
    credentials: str
    language_model: str
    memory_backend: str
    message_broker: str
    mind: str
    plugin_manager: str
    workspace: str

############################
####### Mind Schema ########
############################
class BasePlanStep(pydantic.BaseModel):
    """A class for plan steps."""
    name: str
    prompt: str

class BaseStepResults(pydantic.BaseModel):
    """A class for plan step results."""
    prompt: str
    variables: Dict[str, str]

class BasePlan(pydantic.BaseModel):
    """A class for plans."""
    steps: List[BasePlanStep]
    deliverables: List[str]

class BaseGoal(pydantic.BaseModel):
    """A class for goals."""
    name: str
    objectives: List[str]

class BaseIntrospection(pydantic.BaseModel):
    """A class for introspection on a plan."""
    plan_efficiency: float
    plan_effectiveness: float
    success_probability: float


class BaseMind(pydantic.BaseModel):
    """
    The mind is the core of the agent. It is responsible for planning, executing, and introspecting on a goal.

    params:
        goal: The goal to achieve.
        plan: The plan to achieve the goal, as calculated by the agent.
        introspection: The introspection on the plan, as calculated by the agent.

        prompt_loader: The prompt loader to use to load the planning, step execution and introspection prompts

        id: The id of the mind.
        name: The name of the mind.
        description: The description of the mind.

        memory: The memory system for the mind.
        workspace: The workspace system for the mind.
        message_broker: The message broker for the mind.
    
    methods:

        plan: Generates a plan given a goal loading the prompt using the prompt loader.
        execute: Executes a plan, the details of which are implementation specific.
        introspect: Introspects on a plan given a goal loading the prompt using the prompt loader.
        spawn_sub_mind: Spawns a sub mind.
    """
    goal: BaseGoal
    plan: Optional[BasePlan] = None

    prompt_loader: BasePromptLoader

    id: str
    name: str
    description: str

    memory: str
    workspace: str
    message_broker: "BaseMessageBroker"

    sub_minds: List["BaseMind"] = []

    async def plan(self) -> BasePlan:
        """Implement logic to generate a plan given a goal."""
        raise NotImplementedError

    async def execute(self) -> None:
        """Implement logic to execute a plan."""
        raise NotImplementedError

    async def introspect(self) -> BaseIntrospection:
        """Implement logic to introspect on a plan given a goal."""
        raise NotImplementedError

    async def spawn_sub_mind(self, goal: BaseGoal, channel_id: str) -> None:
        """Spawns a sub mind with the given goal and channel id."""
        raise NotImplementedError

############################
##### Memory Schema ########
############################
## copied from @Pwuts work #
############################

Embedding = list[np.float32] | np.ndarray[Any, np.dtype[np.float32]]
MemoryDocType = Literal["webpage", "text_file", "code_file", "agent_history"]

class Message(pydantic.BaseModel):
    """OpenAI Message object containing a role and the message content"""
    role: str
    content: str

class MemoryItem(pydantic.BaseModel):
    """Memory object containing raw content as well as embeddings"""

    raw_content: str
    summary: str
    chunks: list[str]
    chunk_summaries: list[str]
    e_summary: Embedding
    e_chunks: list[Embedding]
    metadata: dict

    async def get_relevance(self, query: str, e_query: Embedding) -> "MemoryItemRelevance":
        """Get the relevance of a memory item to a query"""
        raise NotImplementedError
    
    @staticmethod
    async def from_text(
        text: str,
        source_type: MemoryDocType,
        metadata: Optional[dict] = None,
        how_to_summarize: Optional[str] = None,
        question_for_summary: Optional[str] = None,
    ) -> "MemoryItem":
        """Create a memory item from text"""
        raise NotImplementedError
    
    @staticmethod
    async def from_text_file(content: str, path: str) -> "MemoryItem":
        """Create a memory item from a text file"""
        return MemoryItem.from_text(content, "text_file", {"location": path})

    @staticmethod
    async def from_code_file(content: str, path: str) -> "MemoryItem":
        """Create a memory item from a code file"""
        # TODO: implement tailored code memories
        return MemoryItem.from_text(content, "code_file", {"location": path})

    @staticmethod
    async def from_ai_action(ai_message: Message, result_message: Message) -> "MemoryItem":
        """
        The result_message contains either user feedback
        or the result of the command specified in ai_message
        """
        return NotImplementedError
    
    @staticmethod
    async def from_webpage(content: str, url: str, question: Optional[str]= None) -> "MemoryItem":
        """Create a memory item from a webpage"""
        return MemoryItem.from_text(
            text=content,
            source_type="webpage",
            metadata={"location": url},
            question_for_summary=question,
        )

    def __str__(self) -> str:
        """String representation of a memory item"""
        raise NotImplementedError

class MemoryItemRelevance:
    """
    Class that encapsulates memory relevance search functionality and data.
    Instances contain a MemoryItem and its relevance scores for a given query.
    """

    memory_item: MemoryItem
    for_query: str
    summary_relevance_score: float
    chunk_relevance_scores: list[float]

    @staticmethod
    def of(
        memory_item: MemoryItem, for_query: str, e_query: Embedding | None = None
    ) -> "MemoryItemRelevance":
        """Create a MemoryItemRelevance instance for a given query"""
        return NotImplementedError

    @staticmethod
    def calculate_scores(
        memory: MemoryItem, compare_to: Embedding
    ) -> tuple[float, float, list[float]]:
        """
        Calculates similarity between given embedding and all embeddings of the memory
        Returns:
            float: the aggregate (max) relevance score of the memory
            float: the relevance score of the memory summary
            list: the relevance scores of the memory chunks
        """
        return NotImplementedError

    @property
    def score(self) -> float:
        """The aggregate relevance score of the memory item for the given query"""
        return NotImplementedError

    @property
    def most_relevant_chunk(self) -> tuple[str, float]:
        """The most relevant chunk of the memory item + its score for the given query"""
        return NotImplementedError

    def __str__(self):
        return (
            f"{self.memory_item.summary} ({self.summary_relevance_score}) "
            f"{self.chunk_relevance_scores}"
        )



class BaseMemoryProvider(pydantic.BaseModel):
    """Base class for memory systems."""
        
    def get(self, query: str) -> Optional[MemoryItemRelevance]:
        """
        Gets the data from the memory that is most relevant to the given query.
        Args:
            data: The data to compare to.
        Returns: The most relevant Memory
        """
        result = self.get_relevant(query, 1)
        return result[0] if result else None

    def get_relevant(self, query: str, k: int) -> Sequence[MemoryItemRelevance]:
        """
        Returns the top-k most relevant memories for the given query
        Args:
            query: the query to compare stored memories to
            k: the number of relevant memories to fetch
        Returns:
            list[MemoryItemRelevance] containing the top [k] relevant memories
        """
        if len(self) < 1:
            return []
        relevances = self.score_memories_for_relevance(query)

        # take last k items and reverse
        top_k_indices = np.argsort([r.score for r in relevances])[-k:][::-1]
        return [relevances[i] for i in top_k_indices]

    def score_memories_for_relevance(
        self, for_query: str,
    ) -> Sequence[MemoryItemRelevance]:
        """
        Returns MemoryItemRelevance for every memory in the index.
        Implementations may override this function for performance purposes.
        """
        return NotImplementedError

    def get_stats(self) -> tuple[int, int]:
        """
        Returns:
            tuple (n_memories: int, n_chunks: int): the stats of the memory index
        """
        return NotImplementedError



############################
### Prompt Loader Schema ###
############################

class BasePromptLoader(pydantic.BaseModel):
    """A base class for prompt loaders."""
    directory: str

    def load_prompt(self, prompt_name: str) -> str:
        """Loads a prompt."""
        raise NotImplementedError

############################
### LLM Provider Schema ####
############################

class BaseLLMProvider(pydantic.BaseModel):
    """The base class for all LLM providers."""

    num_retries: int = (10,)
    debug_mode: bool = (False,)

    async def create_chat_completion(self, messages: list) -> str:
        """Create a chat completion"""
        raise NotImplementedError
    

class BaseAIModel(pydantic.BaseModel):
    """
    Base class for all models.
    """





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
    mind: BaseMind

    async def run(self) -> None:
        """Runs the agent"""
        raise NotImplementedError

    async def stop(self) -> None:
        """Stops the agent"""
        raise NotImplementedError
