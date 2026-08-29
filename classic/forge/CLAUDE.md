# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

All commands run from the `classic/` directory (parent of this directory):

```bash
# Run forge agent server (port 8000)
poetry run python -m forge

# Run forge tests
poetry run pytest forge/tests/
poetry run pytest forge/tests/ --cov=forge
poetry run pytest -k test_name
```

## Entry Point

`__main__.py` → loads `.env` → configures logging → starts Uvicorn with hot-reload on port 8000

The app is created in `app.py`:
```python
agent = ForgeAgent(database=database, workspace=workspace)
app = agent.get_agent_app()
```

## Directory Structure

```
forge/
├── __main__.py               # Entry: uvicorn server startup
├── app.py                    # FastAPI app creation
├── agent/                    # Core agent framework
│   ├── base.py               # BaseAgent abstract class
│   ├── forge_agent.py        # Reference implementation
│   ├── components.py         # AgentComponent base classes
│   └── protocols.py          # Protocol interfaces
├── agent_protocol/           # Agent Protocol standard
│   ├── agent.py              # ProtocolAgent mixin
│   ├── api_router.py         # FastAPI routes
│   └── database/             # Task/step persistence
├── command/                  # Command system
│   ├── command.py            # Command class
│   ├── decorator.py          # @command decorator
│   └── parameter.py          # CommandParameter
├── components/               # Built-in components
│   ├── action_history/       # Track & summarize actions
│   ├── code_executor/        # Python & shell execution
│   ├── context/              # File/folder context
│   ├── file_manager/         # File operations
│   ├── git_operations/       # Git commands
│   ├── image_gen/            # DALL-E & SD
│   ├── system/               # Core directives + finish
│   ├── user_interaction/     # User prompts
│   ├── watchdog/             # Loop detection
│   └── web/                  # Search & Selenium
├── config/                   # Configuration models
├── llm/                      # LLM integration
│   └── providers/            # OpenAI, Anthropic, Groq, etc.
├── file_storage/             # Storage abstraction
│   ├── base.py               # FileStorage ABC
│   ├── local.py              # LocalFileStorage
│   ├── s3.py                 # S3FileStorage
│   └── gcs.py                # GCSFileStorage
├── models/                   # Core data models
├── content_processing/       # Text/HTML utilities
├── logging/                  # Structured logging
└── json/                     # JSON parsing utilities
```

## Core Abstractions

### BaseAgent (`agent/base.py`)

Abstract base for all agents. Generic over proposal type.

```python
class BaseAgent(Generic[AnyProposal], metaclass=AgentMeta):
    def __init__(self, settings: BaseAgentSettings)
```

**Must Override:**
```python
async def propose_action(self) -> AnyProposal
async def execute(self, proposal: AnyProposal, user_feedback: str) -> ActionResult
async def do_not_execute(self, denied_proposal: AnyProposal, user_feedback: str) -> ActionResult
```

**Key Methods:**
```python
async def run_pipeline(protocol_method, *args, retry_limit=3) -> list
# Executes protocol across all matching components with retry logic

def dump_component_configs(self) -> str  # Serialize configs to JSON
def load_component_configs(self, json: str)  # Restore configs
```

**Configuration (`BaseAgentConfiguration`):**
```python
fast_llm: ModelName = "gpt-3.5-turbo-16k"
smart_llm: ModelName = "gpt-4"
big_brain: bool = True              # Use smart_llm
cycle_budget: Optional[int] = 1     # Steps before approval needed
send_token_limit: Optional[int]     # Prompt token budget
```

### Component System (`agent/components.py`)

**AgentComponent** - Base for all components:
```python
class AgentComponent(ABC):
    _run_after: list[type[AgentComponent]] = []
    _enabled: bool | Callable[[], bool] = True
    _disabled_reason: str = ""

    def run_after(self, *components) -> Self  # Set execution order
    def enabled(self) -> bool                  # Check if active
```

**ConfigurableComponent** - Components with Pydantic config:
```python
class ConfigurableComponent(Generic[BM]):
    config_class: ClassVar[type[BM]]  # Set in subclass

    @property
    def config(self) -> BM  # Get/create config from env
```

**Component Discovery:**
1. Agent assigns components: `self.foo = FooComponent()`
2. `AgentMeta.__call__` triggers `_collect_components()`
3. Components are topologically sorted by `run_after` dependencies
4. Disabled components skipped during pipeline execution

### Protocols (`agent/protocols.py`)

Protocols define what components CAN do:

```python
class DirectiveProvider(AgentComponent):
    def get_constraints(self) -> Iterator[str]
    def get_resources(self) -> Iterator[str]
    def get_best_practices(self) -> Iterator[str]

class CommandProvider(AgentComponent):
    def get_commands(self) -> Iterator[Command]

class MessageProvider(AgentComponent):
    def get_messages(self) -> Iterator[ChatMessage]

class AfterParse(AgentComponent, Generic[AnyProposal]):
    def after_parse(self, result: AnyProposal) -> None

class AfterExecute(AgentComponent):
    def after_execute(self, result: ActionResult) -> None

class ExecutionFailure(AgentComponent):
    def execution_failure(self, error: Exception) -> None
```

**Pipeline execution:**
```python
results = await self.run_pipeline(CommandProvider.get_commands)
# Iterates all components implementing CommandProvider
# Collects all yielded Commands
# Handles retries on ComponentEndpointError
```

## LLM Providers (`llm/providers/`)

### MultiProvider

Routes to correct provider based on model name:

```python
class MultiProvider:
    async def create_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        model_name: ModelName,
        **kwargs
    ) -> ChatModelResponse

    async def get_available_chat_models(self) -> Sequence[ChatModelInfo]
```

### Supported Models

```python
# OpenAI
OpenAIModelName.GPT3, GPT3_16k, GPT4, GPT4_32k, GPT4_TURBO, GPT4_O

# Anthropic
AnthropicModelName.CLAUDE3_OPUS, CLAUDE3_SONNET, CLAUDE3_HAIKU
AnthropicModelName.CLAUDE3_5_SONNET, CLAUDE3_5_SONNET_v2, CLAUDE3_5_HAIKU
AnthropicModelName.CLAUDE4_SONNET, CLAUDE4_OPUS, CLAUDE4_5_OPUS

# Groq
GroqModelName.LLAMA3_8B, LLAMA3_70B, MIXTRAL_8X7B
```

### Key Types

```python
class ChatMessage(BaseModel):
    role: Role  # USER, SYSTEM, ASSISTANT, TOOL, FUNCTION
    content: str

class AssistantFunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any]

class ChatModelResponse(BaseModel):
    completion_text: str
    function_calls: list[AssistantFunctionCall]
```

## File Storage (`file_storage/`)

Abstract interface for file operations:

```python
class FileStorage(ABC):
    def open_file(self, path, mode="r", binary=False) -> IO
    def read_file(self, path, binary=False) -> str | bytes
    async def write_file(self, path, content) -> None
    def list_files(self, path=".") -> list[Path]
    def list_folders(self, path=".", recursive=False) -> list[Path]
    def delete_file(self, path) -> None
    def exists(self, path) -> bool
    def clone_with_subroot(self, subroot) -> FileStorage
```

**Implementations:** `LocalFileStorage`, `S3FileStorage`, `GCSFileStorage`

## Command System (`command/`)

### @command Decorator

```python
@command(
    names=["greet", "hello"],
    description="Greet a user",
    parameters={
        "name": JSONSchema(type=JSONSchema.Type.STRING, required=True),
        "greeting": JSONSchema(type=JSONSchema.Type.STRING, required=False),
    },
)
def greet(self, name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"
```

### Providing Commands

```python
class MyComponent(CommandProvider):
    def get_commands(self) -> Iterator[Command]:
        yield self.greet  # Decorated method becomes Command
```

## Built-in Components

| Component | Protocols | Purpose |
|-----------|-----------|---------|
| `SystemComponent` | DirectiveProvider, MessageProvider, CommandProvider | Core directives, `finish` command |
| `FileManagerComponent` | DirectiveProvider, CommandProvider | read/write/list files |
| `CodeExecutorComponent` | CommandProvider | Python & shell execution (Docker) |
| `WebSearchComponent` | DirectiveProvider, CommandProvider | DuckDuckGo & Google search |
| `WebPlaywrightComponent` | DirectiveProvider, CommandProvider | Browser automation (Playwright) |
| `ActionHistoryComponent` | MessageProvider, AfterParse, AfterExecute | Track & summarize history |
| `WatchdogComponent` | AfterParse | Loop detection, LLM switching |
| `ContextComponent` | MessageProvider, CommandProvider | Keep files in prompt context |
| `ImageGeneratorComponent` | CommandProvider | DALL-E, Stable Diffusion |
| `GitOperationsComponent` | CommandProvider | Git commands |
| `UserInteractionComponent` | CommandProvider | `ask_user` command |

## Configuration

### BaseAgentSettings

```python
class BaseAgentSettings(SystemSettings):
    agent_id: str
    ai_profile: AIProfile          # name, role, goals
    directives: AIDirectives       # constraints, resources, best_practices
    task: str
    config: BaseAgentConfiguration
```

### UserConfigurable Fields

```python
class MyConfig(SystemConfiguration):
    api_key: SecretStr = UserConfigurable(from_env="API_KEY", exclude=True)
    max_retries: int = UserConfigurable(default=3, from_env="MAX_RETRIES")

config = MyConfig.from_env()  # Load from environment
```

## Agent Protocol (`agent_protocol/`)

REST API for task-based interaction:

```
POST /ap/v1/agent/tasks              # Create task
GET  /ap/v1/agent/tasks              # List tasks
GET  /ap/v1/agent/tasks/{id}         # Get task
POST /ap/v1/agent/tasks/{id}/steps   # Execute step
GET  /ap/v1/agent/tasks/{id}/steps   # List steps
GET  /ap/v1/agent/tasks/{id}/artifacts  # List artifacts
```

**ProtocolAgent mixin** provides these endpoints + database persistence.

## Testing

**Fixtures** (`conftest.py`):
- `storage` - Temporary LocalFileStorage

Run from the `classic/` directory:
```bash
poetry run pytest forge/tests/                    # All forge tests
poetry run pytest forge/tests/ --cov=forge        # With coverage
```

**Note**: Tests requiring API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY) will be skipped if not set.

## Creating a Custom Component

```python
from forge.agent.components import AgentComponent, ConfigurableComponent
from forge.agent.protocols import CommandProvider
from forge.command import command
from forge.models.json_schema import JSONSchema

class MyConfig(BaseModel):
    setting: str = "default"

class MyComponent(CommandProvider, ConfigurableComponent[MyConfig]):
    config_class = MyConfig

    def get_commands(self) -> Iterator[Command]:
        yield self.my_command

    @command(
        names=["mycmd"],
        description="Do something",
        parameters={"arg": JSONSchema(type=JSONSchema.Type.STRING, required=True)},
    )
    def my_command(self, arg: str) -> str:
        return f"Result: {arg}"
```

## Creating a Custom Agent

```python
from forge.agent.forge_agent import ForgeAgent

class MyAgent(ForgeAgent):
    def __init__(self, database, workspace):
        super().__init__(database, workspace)
        self.my_component = MyComponent()

    async def propose_action(self) -> ActionProposal:
        # 1. Collect directives
        constraints = await self.run_pipeline(DirectiveProvider.get_constraints)
        resources = await self.run_pipeline(DirectiveProvider.get_resources)

        # 2. Collect commands
        commands = await self.run_pipeline(CommandProvider.get_commands)

        # 3. Collect messages
        messages = await self.run_pipeline(MessageProvider.get_messages)

        # 4. Build prompt and call LLM
        response = await self.llm_provider.create_chat_completion(
            model_prompt=messages,
            model_name=self.config.smart_llm,
            functions=function_specs_from_commands(commands),
        )

        # 5. Parse and return proposal
        return ActionProposal(
            thoughts=response.completion_text,
            use_tool=response.function_calls[0],
            raw_message=AssistantChatMessage(content=response.completion_text),
        )
```

## Key Patterns

### Component Ordering
```python
self.component_a = ComponentA()
self.component_b = ComponentB().run_after(self.component_a)
```

### Conditional Enabling
```python
self.search = WebSearchComponent()
self.search._enabled = bool(os.getenv("GOOGLE_API_KEY"))
self.search._disabled_reason = "No Google API key"
```

### Pipeline Retry Logic
- `ComponentEndpointError` → retry same component (3x)
- `EndpointPipelineError` → restart all components (3x)
- `ComponentSystemError` → restart all pipelines

## Key Files Reference

| Purpose | Location |
|---------|----------|
| Entry point | `__main__.py` |
| FastAPI app | `app.py` |
| Base agent | `agent/base.py` |
| Reference agent | `agent/forge_agent.py` |
| Components base | `agent/components.py` |
| Protocols | `agent/protocols.py` |
| LLM providers | `llm/providers/` |
| File storage | `file_storage/` |
| Commands | `command/` |
| Built-in components | `components/` |
| Agent Protocol | `agent_protocol/` |
