from importlib import import_module

from autogpts.autogpt.autogpt.core.configuration import SystemConfiguration

# from autogpts.autogpt.autogpt.core.workspace.simple import SimpleWorkspace


class BaseAgentSystems(SystemConfiguration):
    memory: str = "autogpt.core.memory.base.Memory"
    workspace: str = "autogpt.core.workspace.SimpleWorkspace"

    class Config(SystemConfiguration.Config):
        extra = "allow"

    @classmethod
    # def load_from_import_path(cls, attr) -> "Configurable":
    def load_from_import_path(cls, system_location: str):
        """Load a plugin from an import path."""
        module_path, _, class_name = system_location.rpartition(".")
        return getattr(import_module(module_path), class_name)


class BaseAgentConfiguration(SystemConfiguration):
    cycle_count: int = 0
    max_task_cycle_count: int = 3
    systems: BaseAgentSystems = BaseAgentSystems()

    class Config(SystemConfiguration.Config):
        extra = "allow"


class BaseAgentDirectives(dict):
    """An object that contains the basic directives for the AI prompt.

    Attributes:
        constraints (list): A list of constraints that the AI should adhere to.
        resources (list): A list of resources that the AI can utilize.
        best_practices (list): A list of best practices that the AI should follow.
    """

    constraints: list[str]
    resources: list[str]
    best_practices: list[str]
