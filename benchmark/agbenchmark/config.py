import json
from pathlib import Path

from agbenchmark.utils.data_types import AgentBenchmarkConfig


def load_agbenchmark_config() -> AgentBenchmarkConfig:
    """
    Loads the AgentBenchmarkConfig from ./agbenchmark_config/config.json.

    Returns:
        AgentBenchmarkConfig: The loaded config object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        JSONDecodeError: If the config file does not contain valid JSON.
    """
    agent_benchmark_config_path = Path.cwd() / "agbenchmark_config" / "config.json"
    with open(agent_benchmark_config_path, "r") as f:
        agent_benchmark_config = AgentBenchmarkConfig(**json.load(f))
        agent_benchmark_config.agent_benchmark_config_path = agent_benchmark_config_path
        return agent_benchmark_config
