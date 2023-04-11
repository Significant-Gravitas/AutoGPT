import dataclasses
import orjson
from typing import Any, List, Optional, Dict
import numpy as np
import os
from memory.base import MemoryProviderSingleton, get_ada_embedding


EMBED_DIM = 1536
SAVE_OPTIONS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS


def create_default_embeddings():
    return np.zeros((0, EMBED_DIM)).astype(np.float32)


@dataclasses.dataclass
class NamespaceContent:
    texts: List[str] = dataclasses.field(default_factory=list)
    embeddings: np.ndarray = dataclasses.field(
        default_factory=create_default_embeddings
    )


@dataclasses.dataclass
class CacheContent:
    cache: Dict[str, NamespaceContent] = dataclasses.field(default_factory=dict)


class LocalCache(MemoryProviderSingleton):

    # on load, load our database
    def __init__(self, cfg) -> None:
        self.filename = f"{cfg.memory_index}.json"
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'rb') as f:
                    loaded = orjson.loads(f.read())
                    self.data = CacheContent(**loaded)
            except Exception as e:
                print("Error loading local cache: ", e)
                print("Creating new cache")
                self.data = CacheContent()
        else:
            self.data = CacheContent()

    def add(self, text: str, namespace="default"):
        """
        Add text to our list of texts, add embedding as row to our
            embeddings-matrix

        Args:
            text: str

        Returns: None
        """
        if 'Command Error:' in text:
            return ""
        
        if namespace not in self.data.cache:
            self.data.cache[namespace] = NamespaceContent()

        self.data.cache[namespace].texts.append(text)

        embedding = get_ada_embedding(text)

        vector = np.array(embedding).astype(np.float32)
        vector = vector[np.newaxis, :]
        self.data.cache[namespace].embeddings = np.concatenate(
            [
                vector,
                self.data.cache[namespace].embeddings,
            ],
            axis=0,
        )

        with open(self.filename, 'wb') as f:
            out = orjson.dumps(
                self.data,
                option=SAVE_OPTIONS
            )
            f.write(out)
        return text

    def clear(self) -> str:
        """
        Clears the redis server.

        Returns: A message indicating that the memory has been cleared.
        """
        self.data = CacheContent()
        return "Obliviated"

    def get(self, data: str, namespace="default") -> Optional[List[Any]]:
        """
        Gets the data from the memory that is most relevant to the given data.

        Args:
            data: The data to compare to.

        Returns: The most relevant data.
        """
        return self.get_relevant(data, 1, namespace=namespace)

    def get_relevant(self, text: str, k: int, namespace="default") -> List[Any]:
        """"
        matrix-vector mult to find score-for-each-row-of-matrix
         get indices for top-k winning scores
         return texts for those indices
        Args:
            text: str
            k: int

        Returns: List[str]
        """
        if namespace not in self.data.cache:
            self.data.cache[namespace] = NamespaceContent()

        embedding = get_ada_embedding(text)

        scores = np.dot(self.data.cache[namespace].embeddings, embedding)

        top_k_indices = np.argsort(scores)[-k:][::-1]

        return [self.data.cache[namespace].texts[i] for i in top_k_indices]

    def get_stats(self):
        """
        Returns: The stats of the local cache.
        """
        num_namespaces = len(self.data.cache)
        namespace_stats = []
        for namespace, content in self.data.cache.items():
            namespace_stats.append(
                {
                    "namespace": namespace,
                    "len": len(content.texts),
                    "shape": content.embeddings.shape,
                }
            )
        return num_namespaces, namespace_stats
