"""The base class for all memory providers."""
from autogpt.agent.memory.base_memory_provider import BaseMemoryProvider


class LocalMemoryProvider(BaseMemoryProvider):
    """The base class for all memory providers."""

    def __init__(self) -> None:
        self.memory_store = {}
        self.objective = None
        self.goals = []
        self.deliverables = []
