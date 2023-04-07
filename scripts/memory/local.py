import dataclasses
import orjson
from typing import Any, List, Optional
import numpy as np
import os
from memory.base import MemoryProviderSingleton, get_ada_embedding


EMBED_DIM = 1536
SAVE_OPTIONS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS


def create_default_embeddings():
    return np.zeros((0, EMBED_DIM)).astype(np.float32)


@dataclasses.dataclass
class CacheContent:
    texts: List[str] = dataclasses.field(default_factory=list)
    embeddings: np.ndarray = dataclasses.field(
        default_factory=create_default_embeddings
    )


class LocalCache(MemoryProviderSingleton):

    # on load, load our database
    def __init__(self, cfg) -> None:
        self.filename = f"{cfg.memory_index}.json"
        if os.path.exists(self.filename):
            with open(self.filename, 'rb') as f:
                loaded = orjson.loads(f.read())
                self.data = CacheContent(**loaded)
        else:
            self.data = CacheContent()

    def add(self, text: str):
        """
        Add text to our list of texts, add embedding as row to our
            embeddings-matrix

        Args:
            text: str

        Returns: None
        """
        self.data.texts.append(text)

        embedding = get_ada_embedding(text)

        vector = np.array(embedding).astype(np.float32)
        vector = vector[np.newaxis, :]
        self.data.embeddings = np.concatenate(
            [
                vector,
                self.data.embeddings,
            ],
            axis=0,
        )

        with open(self.filename, 'wb') as f:
            out = orjson.dumps(
                self.data,
                option=SAVE_OPTIONS
            )
            f.write(out)

    def clear(self) -> str:
        """
        Clears the redis server.

        Returns: A message indicating that the memory has been cleared.
        """
        self.data = CacheContent()
        return "Obliviated"

    def get(self, data: str) -> Optional[List[Any]]:
        """
        Gets the data from the memory that is most relevant to the given data.

        Args:
            data: The data to compare to.

        Returns: The most relevant data.
        """
        return self.get_relevant(data, 1)

    def get_relevant(self, text: str, k: int) -> List[Any]:
        """"
        matrix-vector mult to find score-for-each-row-of-matrix
         get indices for top-k winning scores
         return texts for those indices
        Args:
            text: str
            k: int

        Returns: List[str]
        """
        embedding = get_ada_embedding(text)

        scores = np.dot(self.data.embeddings, embedding)

        top_k_indices = np.argsort(scores)[-k:][::-1]

        return [self.data.texts[i] for i in top_k_indices]

    def get_stats(self):
        """
        Returns: The stats of the local cache.
        """
        return len(self.data.texts), self.data.embeddings.shape
