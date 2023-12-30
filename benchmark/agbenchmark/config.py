import json

from agbenchmark.utils.data_types import AgentBenchmarkConfig
from agbenchmark.utils.path_manager import PATH_MANAGER


def load_agbenchmark_config() -> AgentBenchmarkConfig:
    """
    Loads the AgentBenchmarkConfig from ./agbenchmark_config/config.json.

    Returns:
        AgentBenchmarkConfig: The loaded config object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        JSONDecodeError: If the config file does not contain valid JSON.
    """
    with open(PATH_MANAGER.config_file, "r") as f:
        agent_benchmark_config = AgentBenchmarkConfig(**json.load(f))
        agent_benchmark_config.agent_benchmark_config_path = PATH_MANAGER.config_file
        return agent_benchmark_config
