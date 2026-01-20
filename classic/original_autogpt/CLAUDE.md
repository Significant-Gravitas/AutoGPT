# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

All commands run from the `classic/` directory (parent of this directory):

```bash
# Run interactive CLI
poetry run autogpt run

# Run Agent Protocol server (port 8000)
poetry run serve --debug

# Run tests
poetry run pytest original_autogpt/tests/
poetry run pytest original_autogpt/tests/unit/ -v
poetry run pytest -k test_name
```

## Entry Points

| Command | Entry | Description |
|---------|-------|-------------|
| `autogpt run` | `app/cli.py:run()` | Interactive agent mode |
| `autogpt serve` | `app/cli.py:serve()` | Agent Protocol server (FastAPI) |

Both ultimately call functions in `app/main.py`:
- `run_auto_gpt()` → `run_interaction_loop(agent)`
- `run_auto_gpt_server()` → Hypercorn + FastAPI

## Directory Structure

```
autogpt/
├── __main__.py                    # Entry: runs cli()
├── app/                           # Application layer
│   ├── cli.py                     # Click CLI (@cli.command decorators)
│   ├── main.py                    # run_auto_gpt(), run_interaction_loop()
│   ├── config.py                  # AppConfig (Pydantic) + ConfigBuilder
│   ├── agent_protocol_server.py   # FastAPI server for Agent Protocol
│   ├── setup.py                   # Interactive AI profile setup
│   └── configurator.py            # Config overrides, model validation
├── agents/                        # Core agent
│   ├── agent.py                   # Agent class (extends BaseAgent)
│   ├── agent_manager.py           # State persistence (load/save)
│   └── prompt_strategies/
│       └── one_shot.py            # Prompt building + response parsing
└── agent_factory/                 # Agent creation
    ├── configurators.py           # create_agent(), configure_agent_with_state()
    └── profile_generator.py       # AI profile generation
```

## Core Architecture

### Agent Class (`agents/agent.py`)

Extends `forge.agent.base.BaseAgent[OneShotAgentActionProposal]`.

**Constructor**:
```python
Agent(
    settings: AgentSettings,      # State: profile, directives, history
    llm_provider: MultiProvider,  # LLM access
    file_storage: FileStorage,    # File access
    app_config: AppConfig,
)
```

**Built-in Components** (initialized in `__init__`):
- `self.system` - System information
- `self.history` - ActionHistoryComponent (episodic memory)
- `self.file_manager` - FileManagerComponent (workspace files)
- `self.code_executor` - CodeExecutorComponent (Docker-based)
- `self.git_ops` - GitOperationsComponent
- `self.image_gen` - ImageGeneratorComponent
- `self.web_search` - WebSearchComponent
- `self.web_browser` - WebPlaywrightComponent
- `self.context` - ContextComponent
- `self.watchdog` - WatchdogComponent
- `self.user_interaction` - UserInteractionComponent

**Key Methods**:
- `propose_action()` → Builds prompt, calls LLM, returns `OneShotAgentActionProposal`
- `execute(proposal)` → Runs the proposed tool, returns `ActionResult`
- `do_not_execute(proposal, feedback)` → Registers user feedback instead

### Main Loop (`app/main.py:run_interaction_loop`)

```
While cycles_remaining > 0:
  1. agent.propose_action() → ActionProposal (thoughts + tool call)
  2. Display thoughts + proposed command to user
  3. Get user feedback (or auto-execute in continuous mode)
  4. agent.execute(proposal) or agent.do_not_execute(proposal, feedback)
  5. Decrement cycles, handle Ctrl+C gracefully
```

**Cycle Budget**:
- Normal mode: `cycles = 1` (prompt user each step)
- Continuous mode: `cycles = continuous_limit or ∞`
- User can extend: "y -5" gives 5 more cycles

### Prompt Strategy (`agents/prompt_strategies/one_shot.py`)

**`OneShotAgentActionProposal`**:
```python
thoughts: AssistantThoughts  # observations, reasoning, plan, self_criticism
use_tool: AssistantFunctionCall  # {name, arguments}
```

**`AssistantThoughts`**:
```python
observations: str      # From last action result
text: str              # Main thoughts
reasoning: str         # Why this thought
self_criticism: str    # Constructive critique
plan: list[str]        # Multi-step plan
speak: str             # What to say to user
```

**Prompt Structure**:
1. System prompt (intro + profile + directives + commands)
2. Task as user message
3. Message history from components
4. "Determine next action" instruction

### Configuration (`app/config.py`)

**`AppConfig`** (Pydantic BaseModel):
```python
smart_llm: ModelName = "gpt-4-turbo"    # Complex reasoning
fast_llm: ModelName = "gpt-3.5-turbo"   # Fast operations
temperature: float = 0.0
continuous_mode: bool = False
continuous_limit: int = 0
restrict_to_workspace: bool = True       # Sandbox file access
disabled_commands: list[str] = []
```

**`ConfigBuilder.build_config_from_env()`** loads from:
1. Hardcoded defaults
2. Environment variables
3. `.env` file
4. CLI arguments (highest priority)

### State Persistence

**Workspace Structure**:
```
data/agents/{agent_id}/
├── state.json          # AgentSettings (profile, directives, history)
└── workspace/          # Agent's working directory
```

**`AgentSettings`** contains:
- `agent_id`, `task`
- `ai_profile` (name, role, goals)
- `ai_directives` (constraints, resources, best practices)
- `history` (EpisodicActionHistory)

**`AgentManager`**:
- `list_agents()` - All agent IDs
- `load_agent_state(agent_id)` - Load from state.json
- `save_state()` - Persist current state

## Memory System

**Short-term** (within execution):
- `agent.event_history` (EpisodicActionHistory)
- Each action creates an `Episode` with action + result
- Token-limited: oldest episodes dropped when limit exceeded

**Long-term** (across sessions):
- Serialized to `state.json` via Pydantic
- Resume with `AgentManager.load_agent_state()`

## Component System

Components implement protocols from forge:
- `CommandProvider.get_commands()` - Provide available commands
- `DirectiveProvider.get_*()` - Provide constraints/resources/best practices
- `MessageProvider.get_messages()` - Provide context messages

**Execution**: `agent.run_pipeline(Protocol.method)` runs all component implementations.

**Ordering**: `component.run_after(other)` controls execution order.

## Forge Dependency

Heavy reliance on `forge` package (sibling directory):
- `forge.agent.base.BaseAgent` - Base class
- `forge.llm.providers.MultiProvider` - LLM abstraction
- `forge.file_storage` - File storage backends
- `forge.components.*` - All component implementations
- `forge.models.config` - Configuration models

## Key Gotchas

1. **Component ordering matters** - Use `run_after()` for dependencies
2. **Token limits are critical** - History auto-drops old episodes; large results get truncated
3. **Continuous mode is dangerous** - No user approval between steps
4. **State files grow large** - Full history in state.json
5. **SIGINT handling** - First Ctrl+C stops continuous mode; second exits
6. **Anthropic limitations** - Doesn't support functions API + prefilling

## CLI Options

```bash
autogpt run [OPTIONS]
  -c, --continuous              # No user approval between steps
  -l, --continuous-limit N      # Max steps in continuous mode
  --ai-name NAME                # Override AI name
  --ai-role ROLE                # Override AI role
  --constraint TEXT             # Add constraint (repeatable)
  --resource TEXT               # Add resource (repeatable)
  --best-practice TEXT          # Add best practice (repeatable)
  --component-config-file PATH  # JSON config for components
  --debug                       # Enable debug logging
  --log-level LEVEL             # Set log level
```

## Testing

**Fixtures** (`tests/conftest.py`):
- `app_data_dir` - Temp directory
- `config` - AppConfig with noninteractive_mode=True
- `storage` - LocalFileStorage
- `llm_provider` - MultiProvider
- `agent` - Fully initialized Agent

**Running** (from `classic/` directory):
```bash
poetry run pytest original_autogpt/tests/                    # All tests
poetry run pytest original_autogpt/tests/unit/ -v            # Unit tests
poetry run pytest original_autogpt/tests/integration/        # Integration tests
poetry run pytest -k test_config                             # By name
OPENAI_API_KEY=sk-dummy poetry run pytest original_autogpt/  # With dummy key
```

## Common Tasks

### Add a New Component
1. Create class extending `forge.components.AgentComponent`
2. Implement protocols (e.g., `CommandProvider.get_commands()`)
3. Add to `Agent.__init__()` after `super().__init__()`
4. Use `run_after()` to set execution order

### Disable a Command
```python
config.disabled_commands.append("execute_python")
```

### Custom LLM
```bash
SMART_LLM=gpt-4
FAST_LLM=gpt-3.5-turbo
TEMPERATURE=0.7
```

## Tracing Execution

1. `__main__.py` → `cli()`
2. `cli.py:run()` → `run_auto_gpt()`
3. `main.py:run_auto_gpt()`:
   - Build config from env
   - Set up file storage
   - Load or create agent
   - Call `run_interaction_loop(agent)`
4. `main.py:run_interaction_loop()`:
   - `agent.propose_action()` → LLM call
   - Display to user
   - Get feedback or auto-execute
   - `agent.execute()` or `agent.do_not_execute()`
   - Loop

## Benchmarking

Run performance benchmarks from the `classic/` directory:

```bash
# Run a single test
poetry run direct-benchmark run --tests ReadFile

# Run with specific strategies and models
poetry run direct-benchmark run \
    --strategies one_shot,rewoo \
    --models claude \
    --parallel 4

# Run regression tests only
poetry run direct-benchmark run --maintain

# List available challenges
poetry run direct-benchmark list-challenges
```

See `direct_benchmark/CLAUDE.md` for full documentation on strategies, model presets, and CLI options.
