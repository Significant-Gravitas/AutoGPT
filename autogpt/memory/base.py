"""Base class for memory providers."""
import abc
import openai
from autogpt.config import AbstractSingleton, Config


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
}.get(cfg.memory_embedder, 1536)


def get_embedding(text):
    text = text.replace("\n", " ")

    # Use the embedder specified in the config
    if cfg.memory_embedder == "sbert":
        # sBERT model
        embedding = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu").encode(text, show_progress_bar=False)
    else:
        # Ada model
        model = "text-embedding-ada-002"
        engine = None
        if cfg.use_azure:
            engine = cfg.get_azure_deployment_id_for_model(model)
            model = None

        embedding = openai.Embedding.create(
            input=[text],
            model=model,
            engine=engine,
        )["data"][0]["embedding"]

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
