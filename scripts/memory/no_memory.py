from typing import Optional, List, Any

from scripts.memory.base import MemoryProviderSingleton

class NoMemory(MemoryProviderSingleton):
    def __init__(self):
        """
        Initializes the NoMemory provider.

        Returns: None
        """
        pass

    @staticmethod
    def add(*args, **kwargs) -> str:
        """
        Adds a data point to the memory. No action is taken in NoMemory.

        Returns: An empty string.
        """
        return ""

    @staticmethod
    def get(*args, **kwargs) -> Optional[List[Any]]:
        """
        Gets the data from the memory that is most relevant to the given data.
        NoMemory always returns None.

        Returns: None
        """
        return None

    @staticmethod
    def clear(*args, **kwargs) -> str:
        """
        Clears the memory. No action is taken in NoMemory.

        Returns: An empty string.
        """
        return ""

    @staticmethod
    def get_relevant(*args, **kwargs) -> Optional[List[Any]]:
        """
        Returns all the data in the memory that is relevant to the given data.
        NoMemory always returns None.

        Returns: None
        """
        return None

    @staticmethod
    def get_stats(*args, **kwargs):
        """
        Returns: An empty dictionary as there are no stats in NoMemory.
        """
        return {}
