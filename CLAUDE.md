# CLAUDE.md - AI Assistant Guide for Auto-GPT

This document provides essential context for AI assistants working with the Auto-GPT codebase.

## Project Overview

Auto-GPT is an autonomous AI agent framework built around GPT models (primarily GPT-4). It enables AI agents to perform complex tasks by breaking them down into sub-tasks, executing commands, and learning from results.

- **Language**: Python 3.10+ (supports 3.10, 3.11)
- **Entry Points**: `python -m autogpt` or `python main.py`
- **CLI Framework**: Click

## Repository Structure

```
Auto-GPT/
├── autogpt/                    # Main application package
│   ├── agent/                  # Agent management and execution loop
│   │   ├── agent.py           # Main Agent class with interaction loop
│   │   └── agent_manager.py   # Multi-agent orchestration (Singleton)
│   ├── commands/              # Command implementations (16 commands)
│   │   ├── file_operations.py # File I/O operations
│   │   ├── execute_code.py    # Python code execution
│   │   ├── web_selenium.py    # Web scraping (Selenium)
│   │   ├── web_playwright.py  # Web browsing (Playwright)
│   │   ├── google_search.py   # Search integration
│   │   ├── image_gen.py       # Image generation (DALLE, HuggingFace, SD)
│   │   ├── git_operations.py  # Git operations
│   │   └── ...                # Other command modules
│   ├── config/                # Configuration management
│   │   ├── config.py          # Singleton Config class
│   │   └── ai_config.py       # AI settings file management
│   ├── memory/                # Memory backend implementations
│   │   ├── base.py            # Abstract base class
│   │   ├── local.py           # Local file cache (default)
│   │   ├── pinecone.py        # Pinecone vector DB
│   │   ├── redismem.py        # Redis backend
│   │   ├── milvus.py          # Milvus/Zilliz vector DB
│   │   └── weaviate.py        # Weaviate vector search
│   ├── prompts/               # Prompt generation and templating
│   ├── speech/                # TTS implementations (ElevenLabs, gTTS, etc.)
│   ├── processing/            # Text and HTML processing
│   ├── json_utils/            # JSON parsing and LLM response fixing
│   ├── workspace/             # File system sandbox
│   ├── cli.py                 # Click-based CLI
│   ├── main.py                # run_auto_gpt() function
│   ├── app.py                 # Command execution dispatch
│   ├── chat.py                # LLM communication
│   ├── llm_utils.py           # OpenAI API wrapper
│   └── plugins.py             # Plugin loading system
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── conftest.py            # Pytest fixtures
├── docs/                      # Documentation (MkDocs)
├── plugins/                   # Plugin directory (.zip files)
├── scripts/                   # Utility scripts
└── benchmark/                 # Performance benchmarking
```

## Development Commands

### Running the Application

```bash
# Run Auto-GPT
python -m autogpt

# With options
python -m autogpt -c                    # Continuous mode
python -m autogpt --gpt3only            # GPT-3.5 only mode
python -m autogpt --gpt4only            # GPT-4 only mode
python -m autogpt -m redis              # Use Redis memory backend
python -m autogpt --debug               # Enable debug logging
```

### Testing

```bash
# Run all tests (excluding slow integration tests)
pytest --cov=autogpt --without-integration --without-slow-integration

# Run with coverage report
pytest --cov=autogpt --cov-report term-missing --cov-branch

# Run specific test file
pytest tests/unit/test_file_operations.py

# Run tests matching a pattern
pytest -k "test_read"
```

### Code Formatting

```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Check linting with flake8
flake8

# Install pre-commit hooks
pre-commit install
```

### Docker

```bash
# Build Docker image
docker build -t auto-gpt .

# Run with Docker Compose (includes Redis)
docker-compose up
```

## Code Style Guidelines

### Formatting Rules

- **Black**: Line length 88, Python 3.10 target
- **isort**: Black-compatible profile, trailing commas
- **flake8**: Selective rules (E303, W293, W291, W292, E305, E231, E302)

### Pre-commit Hooks

Pre-commit runs automatically on commit:
1. Large file check (max 500KB)
2. Byte-order-marker check
3. Merge conflict check
4. isort import sorting
5. Black code formatting
6. Pytest (unit tests with coverage)

### Singleton Pattern

The codebase uses singletons for global state:
- `Config` - Configuration management
- `AgentManager` - Multi-agent orchestration

Use the `Singleton` metaclass from `autogpt/singleton.py`:
```python
from autogpt.singleton import Singleton

class MyClass(metaclass=Singleton):
    pass
```

## Testing Patterns

### Test Structure

```python
# tests/unit/test_example.py
import unittest
from autogpt.config import Config
from autogpt.workspace import Workspace

class TestExample(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        # Setup workspace for file operations
        workspace_path = os.path.join(os.path.dirname(__file__), "workspace")
        self.workspace_path = Workspace.make_workspace(workspace_path)
        self.config.workspace_path = workspace_path
        self.workspace = Workspace(workspace_path, restrict_to_workspace=True)

    def tearDown(self):
        # Clean up workspace
        shutil.rmtree(self.workspace_path)

    def test_something(self):
        # Test implementation
        pass
```

### Pytest Fixtures (from conftest.py)

```python
@pytest.fixture()
def workspace(workspace_root: Path) -> Workspace:
    workspace_root = Workspace.make_workspace(workspace_root)
    return Workspace(workspace_root, restrict_to_workspace=True)

@pytest.fixture()
def config(workspace: Workspace) -> Config:
    config = Config()
    old_ws_path = config.workspace_path
    config.workspace_path = workspace.root
    yield config
    config.workspace_path = old_ws_path
```

## Configuration

### Environment Variables (.env)

Key environment variables (see `.env.template` for full list):

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `SMART_LLM_MODEL` | Smart model | gpt-4 |
| `FAST_LLM_MODEL` | Fast model | gpt-3.5-turbo |
| `MEMORY_BACKEND` | Memory type | local |
| `EXECUTE_LOCAL_COMMANDS` | Allow local commands | False |
| `RESTRICT_TO_WORKSPACE` | Sandbox file ops | True |
| `HEADLESS_BROWSER` | Browser mode | True |

### AI Settings (ai_settings.yaml)

```yaml
ai_name: MyAgent
ai_role: A helpful AI assistant
ai_goals:
  - Goal 1
  - Goal 2
```

## Key Architectural Patterns

### Command Registry

Commands are registered using decorators:
```python
from autogpt.commands.command import command

@command(
    "command_name",
    "Command description",
    '"arg1": "<value>", "arg2": "<value>"'
)
def my_command(arg1: str, arg2: str) -> str:
    return "Result"
```

### Memory Backends

All memory backends extend `MemoryProviderSingleton`:
- `get_relevant(text, num_relevant)` - Retrieve relevant memories
- `add(text)` - Store a memory
- `clear()` - Clear all memories

### Workspace Security

The `Workspace` class provides sandboxed file operations:
- Path sanitization and validation
- Null-byte injection protection
- Directory traversal prevention
- Configurable restriction via `RESTRICT_TO_WORKSPACE`

## Plugin System

Plugins extend Auto-GPT functionality:
- Place `.zip` plugin files in `plugins/` directory
- Configure allowed plugins via `ALLOWLISTED_PLUGINS` env var
- Plugins use `AutoGPTPluginTemplate` interface
- Supports pre/post instruction hooks

## Common Development Tasks

### Adding a New Command

1. Create a new file in `autogpt/commands/` or add to existing
2. Use the `@command` decorator
3. Add tests in `tests/unit/test_commands.py`
4. Run tests: `pytest tests/unit/test_commands.py`

### Adding a Memory Backend

1. Create new file in `autogpt/memory/`
2. Extend `MemoryProviderSingleton` base class
3. Implement required methods: `get_relevant`, `add`, `clear`
4. Register in memory selection logic
5. Add integration tests

### Modifying LLM Behavior

- `autogpt/llm_utils.py` - OpenAI API interactions
- `autogpt/chat.py` - Message construction and context
- `autogpt/prompts/` - Prompt templates

## CI/CD Pipeline

GitHub Actions workflows:
- **ci.yml**: Linting (flake8, black, isort) + pytest on Python 3.10, 3.11
- **docker-ci.yml**: Docker builds on push
- **docker-release.yml**: Production Docker releases
- **benchmarks.yml**: Performance testing

## Important Notes for AI Assistants

1. **Config Singleton**: The `Config` class is a singleton. Changes persist across the session.

2. **Workspace Isolation**: File operations should use the `Workspace` class for security.

3. **Pre-commit Required**: Always run `black .` and `isort .` before committing.

4. **Test Coverage**: New features must include tests. PRs are checked by CodeCov.

5. **No New Commands**: New commands should be implemented as plugins, not in core.

6. **Memory Backend Selection**: Default is `local`. For production, consider `redis` or `pinecone`.

7. **Token Limits**: Be aware of token limits (`FAST_TOKEN_LIMIT=4000`, `SMART_TOKEN_LIMIT=8000`).

8. **Environment Setup**: Copy `.env.template` to `.env` and configure before running.

## Dependencies

Core:
- `openai==0.27.2` - OpenAI API client
- `click` - CLI framework
- `python-dotenv==1.0.0` - Environment loading
- `tiktoken==0.3.3` - Token counting
- `selenium==4.1.4` - Web automation
- `beautifulsoup4>=4.12.2` - HTML parsing
- `spacy>=3.0.0,<4.0.0` - NLP processing

Dev:
- `pytest` + plugins (asyncio, cov, mock, vcr)
- `black`, `isort`, `flake8`
- `pre-commit`
- `coverage`
