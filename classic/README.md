# AutoGPT Classic

AutoGPT Classic was an experimental project to demonstrate autonomous GPT-4 operation. It was designed to make GPT-4 independently operate and chain together tasks to achieve more complex goals.

## Project Status

**This project is unsupported, and dependencies will not be updated.** It was an experiment that has concluded its initial research phase. If you want to use AutoGPT, you should use the [AutoGPT Platform](/autogpt_platform).

For those interested in autonomous AI agents, we recommend exploring more actively maintained alternatives or referring to this codebase for educational purposes only.

## Overview

AutoGPT Classic was one of the first implementations of autonomous AI agents - AI systems that can independently:
- Break down complex goals into smaller tasks
- Execute those tasks using available tools and APIs
- Learn from the results and adjust its approach
- Chain multiple actions together to achieve an objective

## Structure

- `/benchmark` - Performance testing tools
- `/forge` - Core autonomous agent framework
- `/original_autogpt` - Original implementation

## Getting Started

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)

### Installation

```bash
# Clone the repository
git clone https://github.com/Significant-Gravitas/AutoGPT.git
cd classic

# Install forge (core library)
cd forge && poetry install

# Or install original_autogpt (includes forge as dependency)
cd original_autogpt && poetry install

# Install benchmark (optional)
cd benchmark && poetry install
```

### Configuration

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
# Edit .env with your OPENAI_API_KEY, etc.
```

### Running

```bash
# Run forge agent
cd forge && poetry run python -m forge

# Run original autogpt server
cd original_autogpt && poetry run serve --debug

# Run autogpt CLI
cd original_autogpt && poetry run autogpt
```

Agents run on `http://localhost:8000` by default.

### Benchmarking

```bash
cd benchmark && poetry run agbenchmark
```

### Testing

```bash
cd forge && poetry run pytest
cd original_autogpt && poetry run pytest
```

## Security Notice

This codebase has **known vulnerabilities** and issues with its dependencies. It will not be updated to new dependencies. Use for educational purposes only.

## License

This project segment is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Documentation

Please refer to the [documentation](https://docs.agpt.co) for more detailed information about the project's architecture and concepts.
