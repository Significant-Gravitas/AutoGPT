"""Base class for memory providers."""
import abc

from autogpt.config import AbstractSingleton, Config
from autogpt.llm_utils import create_embedding_with_ada

cfg = Config()


EMBED_DIM = {
    "ada": 1536,
    "sbert": 768
}.get(cfg.memory_embeder, 1536)


try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    if cfg.memory_embedder == "sbert":
        print("Error: Sentence Transformers is not installed. Please install sentence_transformers to use sBERT as an embedder. Defaulting to Ada.")
        cfg.memory_embeder = "ada"


def get_embedding(text: str) -> list:
    """Get the embedding for a given text.

    Args:
        text (str): Text to get embedding for.

    Returns:
        list: Embedding for the given text.
    """
    text = text.replace("\n", " ")

    if cfg.memory_embeder == "sbert":
        # sBERT model
        embedding = get_sbert_embedding(text)
    else:
        # Ada model
        embedding = create_embedding_with_ada(text)

    return embedding


def get_sbert_embedding(text):
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu").encode(text, show_progress_bar=False)

class MemoryProviderSingleton(AbstractSingleton):
    @abc.abstractmethod
    def add(self, data):
        pass

    @abc.abstractmethod
    def get(self, data):
        pass

    @abc.abstractmethod
    def clear(self):
        pass

    @abc.abstractmethod
    def get_relevant(self, data, num_relevant=5):
        pass

    @abc.abstractmethod
    def get_stats(self):
        pass
