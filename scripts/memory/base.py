"""Base class for memory providers."""
import abc
from config import AbstractSingleton, Config
import openai

# try to import sentence transformers, if it fails, default to ada
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    if cfg.memory_embedder == "sbert":
        print("Error: Sentence Transformers is not installed. Please install sentence_transformers"
            " to use sBERT as an embedder. Defaulting to Ada.")
        cfg.memory_embedder = "ada"


cfg = Config()
# Dimension of embeddings encoded by embedders
EMBED_DIM = {
    "ada": 1536,
    "sbert": 768
}.get(cfg.memory_embedder, default=1536)


def get_embedding(text):
    text = text.replace("\n", " ")

    # use the embedder specified in the config
    if cfg.memory_embedder == "sbert":
        embedding = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu").encode(text, show_progress_bar=False)
    else:
        embedding = openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]
    
    return embedding
    

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
