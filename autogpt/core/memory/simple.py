from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.memory.base import Memory


class MemoryConfiguration(SystemConfiguration):
    pass


class MemorySettings(SystemSettings):
    configuration: MemoryConfiguration


class SimpleMemory(Memory, Configurable):
    defaults = MemorySettings(
        name="simple_memory",
        description="A simple memory.",
        configuration=MemoryConfiguration(),
    )
