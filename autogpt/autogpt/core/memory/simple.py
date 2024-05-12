import json
import logging

from forge.models.config import Configurable, SystemConfiguration, SystemSettings

from autogpt.core.memory.base import Memory
from autogpt.core.workspace import Workspace


class MemoryConfiguration(SystemConfiguration):
    pass


class MemorySettings(SystemSettings):
    configuration: MemoryConfiguration


class MessageHistory:
    def __init__(self, previous_message_history: list[str]):
        self._message_history = previous_message_history


class SimpleMemory(Memory, Configurable):
    default_settings = MemorySettings(
        name="simple_memory",
        description="A simple memory.",
        configuration=MemoryConfiguration(),
    )

    def __init__(
        self,
        settings: MemorySettings,
        logger: logging.Logger,
        workspace: Workspace,
    ):
        self._configuration = settings.configuration
        self._logger = logger
        self._message_history = self._load_message_history(workspace)

    @staticmethod
    def _load_message_history(workspace: Workspace):
        message_history_path = workspace.get_path("message_history.json")
        if message_history_path.exists():
            with message_history_path.open("r") as f:
                message_history = json.load(f)
        else:
            message_history = []
        return MessageHistory(message_history)
