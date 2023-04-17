""" GPTCache memory storage provider."""
import os

from gptcache import Cache, Config
from gptcache.adapter.api import get, put, init_similar_cache

from autogpt.memory.base import MemoryProviderSingleton


def nop_post(messages):
    return messages


class GPTCacheMemory(MemoryProviderSingleton):
    """A class that stores the memory to GPTCache"""

    def _init_gptcache(self):
        """Initialize the GPTCache object.
        For the GPTCache, currently includes the following parameters:
        1. pre_embedding_func, the function for getting the requested text information
        2. embedding_func, the function for embedding, convert text to vector
        3. data_manager, data management, including save, search and evict
        4. similarity_evaluation, the function for determining the similarity between the input request and the cache requests
        5. post_process_messages_func, the function for post-processing cache result
        6. config, configure GPTCache, such as similarity threshold

        At present, the default similar search configuration of GPTCache is used,
        and the config will be read later to support more customization.
        For detailed GPTCache usage, refer to: https://github.com/zilliztech/GPTCache

        Returns:
            None
        """
        data_dir = "api_cache"
        self.cache_files.extend([f"{data_dir}/sqlite.db", f"{data_dir}/faiss.index"])
        init_similar_cache(
            data_dir=data_dir,
            cache_obj=self.gptcache,
            post_func=nop_post,
            config=Config(similarity_threshold=0),
        )

    def __init__(self, cfg) -> None:
        """Initialize a class instance

        Args:
            cfg: Config object

        Returns:
            None
        """
        self.gptcache: Cache = Cache()
        self.cache_files = []
        self.cfg = cfg
        self._init_gptcache()

    def add(self, data) -> str:
        """Add an embedding of data into memory.

        Args:
            data (str): The raw text to construct embedding index.

        Returns:
            str: log.
        """

        put(data, data, cache_obj=self.gptcache)
        return f"Inserting data into GPTCache memory\ndata: {data}"

    def get(self, data):
        """Return the most relevant data in memory.
        Args:
            data: The data to compare to.
        """
        return self.get_relevant(data, 1)

    def clear(self) -> str:
        """Clears the GPTCache store file."""
        for cache_file in self.cache_files:
            if os.path.isfile(cache_file):
                os.remove(cache_file)
        self._init_gptcache()
        return f"Clear all cache files, include: {str(self.cache_files)}"

    def get_relevant(self, data: str, num_relevant: int = 5):
        """Return the top-k relevant data in memory.
        Args:
            data: The data to compare to.
            num_relevant (int, optional): The max number of relevant data.
                Defaults to 5.

        Returns:
            list: The top-k relevant data.
        """
        return get(data, cache_obj=self.gptcache, top_k=num_relevant)

    def get_stats(self) -> str:
        """
        Returns: The stats of the GPTCache.
        """
        return f"Using GPTCache memory"
