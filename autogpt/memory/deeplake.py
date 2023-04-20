from typing import List, Union

import deeplake
from deeplake.constants import MB
import numpy as np

from autogpt.llm_utils import create_embedding_with_ada
from autogpt.memory.base import MemoryProviderSingleton


class DeepLakeMemory(MemoryProviderSingleton):
    """DeepLake Storage Memory Provider"""

    def __init__(self, cfg):
        """Initialize DeepLake Storage Memory Provider"""
        dataset_name = cfg.deeplake_dataset_name
        token = cfg.deeplake_token
        self.dataset = deeplake.empty(dataset_name, token=token)
        self.dataset.create_tensor(
            "raw_text",
            htype="text",
            create_id_tensor=False,
            create_sample_info_tensor=False,
            create_shape_tensor=False,
            chunk_compression="lz4",
        )

        self.dataset.create_tensor(
            "embedding",
            htype="generic",
            dtype=np.float64,
            create_id_tensor=False,
            create_sample_info_tensor=False,
            max_chunk_size=64 * MB,
            create_shape_tensor=True,
        )

    def add(self, text):
        """Add an embedding of data into memory.

        Args:
            data (str): The raw text to construct embedding index.

        Returns:
            str: log.
        """
        embedding = create_embedding_with_ada(text)
        self.dataset.embedding.append(embedding)
        self.dataset.raw_text.append(text)
        return text

    def clear(self):
        self.dataset.delete()
        return "Obliviated"

    def get_relevant(self, text, num_relevant=5) -> List[str]:
        """Return the top-k relevant data in memory.
        Args:
            text: The data to compare to.
            num_relevant (int, optional): The max number of relevant data.
                Defaults to 5.

        Returns:
            list: The top-k relevant data.
        """
        embedding = create_embedding_with_ada(text)
        embeddings = self.dataset.embedding.numpy(fetch_chunks=True)
        scores = np.dot(embeddings, embedding)
        top_k_indices = np.argsort(scores)[-num_relevant:][::-1]

        return [
            self.dataset.raw_text[int(i)].numpy(fetch_chunks=True)
            for i in top_k_indices
        ]

    def get(self, data: str) -> Union[list[str], None]:
        """
        Gets the data from the memory that is most relevant to the given data.

        Args:
            data: The data to compare to.

        Returns: The most relevant data.
        """
        return self.get_relevant(data, 1)

    def get_stats(self):
        return self.dataset.summary()
