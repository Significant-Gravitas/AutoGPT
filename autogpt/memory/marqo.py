"""Marqo memory storage provider."""
from typing import Dict, List

import marqo

from autogpt.config import Config
from autogpt.memory.base import MemoryProviderSingleton


class MarqoMemory(MemoryProviderSingleton):
    """Marqo memory storage provider."""

    def __init__(self, cfg: Config):
        """Construct a marqo memory storage object.

        Args:
            cfg (Config): Auto-GPT global config.
        """
        url = cfg.marqo_url
        api_key = cfg.marqo_api_key

        self._client = marqo.Client(url=url, api_key=api_key)

        self._index = cfg.marqo_index_name

        try:
            self._client.create_index(self._index)
            print(f"Created index {self._index}")
        except Exception:
            print(f"Index {self._index} already exists")

    def add(self, data: str) -> str:
        """Add data to memory.

        Args:
            data (str): The data to add.

        Returns:
            str: A description of the action performed.
        """
        resp = self._client.index(self._index).add_documents([{"data": data}])
        return f"Inserting data into memory with id: {resp['items'][0]['_id']}:\n data: {data}"

    def get(self, data: str) -> List[str]:
        """Get the single most relevant piece of data.

        Args:
            data (str): The data to search with.

        Returns:
            List[str]: A list with the single most relevant piece of data.
        """
        return self.get_relevant(data, 1)

    def clear(self) -> str:
        self._client.delete_index(self._index)
        self._client.create_index(self._index)
        return f"Index {self._index} has been cleared"

    def get_relevant(self, data: str, num_relevant: int = 5) -> List[str]:
        """Get the top num_relevant pieces of data.

        Args:
            data (str): The data to search with.
            num_relevant (int, optional): The number of datas to get. Defaults to 5.

        Returns:
            List[str]: A list of relevant datas (ordered in descending relevancy).
        """
        results = self._client.index(self._index).search(q=data, limit=num_relevant)
        return [res["data"] for res in results["hits"]]

    def get_stats(self) -> Dict[str, int]:
        """Get index stats (number of documents)

        Returns:
            Dict[str, int]: Index stats
        """
        return self._client.index(self._index).get_stats()
