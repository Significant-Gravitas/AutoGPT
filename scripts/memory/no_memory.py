from typing import Optional, List, Any

from memory.base import MemoryProviderSingleton

class NoMemory(MemoryProviderSingleton):
    def __init__(self):
        """
        Initializes the NoMemory provider.

        Args:
            cfg: The config object.

        Returns: None
        """
        pass

    @staticmethod
    def add() -> str:
        """
        Adds a data point to the memory. No action is taken in NoMemory.

        Returns: An empty string.
        """
        return ""

    @staticmethod
    def get() -> Optional[List[Any]]:
        """
        Gets the data from the memory that is most relevant to the given data.
        NoMemory always returns None.

        Args:
            data: The data to compare to.

        Returns: None
        """
        return None

    @staticmethod
    def clear() -> str:
        """
        Clears the memory. No action is taken in NoMemory.

        Returns: An empty string.
        """
        return ""

    @staticmethod
    def get_relevant() -> Optional[List[Any]]:
        """
        Returns all the data in the memory that is relevant to the given data.
        NoMemory always returns None.

        Args:
            data: The data to compare to.
            num_relevant: The number of relevant data to return.

        Returns: None
        """
        return None

    @staticmethod
    def get_stats():
        """
        Returns: An empty dictionary as there are no stats in NoMemory.
        """
        return {}
