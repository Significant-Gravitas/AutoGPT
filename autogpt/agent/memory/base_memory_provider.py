"""The base class for all memory providers."""
import abc


class BaseMemoryProvider(abc.ABC):
    """The base class for all memory providers."""

    def __init__(self) -> None:
        self.memory_store = {}
        self.objective = None
        self.goals = []
        self.deliverables = []
