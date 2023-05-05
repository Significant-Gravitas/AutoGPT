import qdrant_client

from autogpt.llm import get_ada_embedding
from autogpt.memory.base import MemoryProviderSingleton


class QdrantMemory(MemoryProviderSingleton):
    # Initializing Qdrant with provided host, port and collection name
    def __init__(self, cfg):
        qdrant_host = cfg.qdrant_host
        qdrant_api_key = cfg.qdrant_api_key
        qdrant_collection_name = cfg.qdrant_collection_name

        self.index = qdrant_client.Index(
            host=qdrant_host,
            api_key=qdrant_api_key,
            collection_name=qdrant_collection_name,
        )

    # add a new data point to the Qdrant collection
    def add(self, data):
        vector = get_ada_embedding(data)  # converts the data into embeddings
        result = self.index.add_entities(
            [vector], ids=[str(self.index.get_collection_size())]
        )  # store the embeddings to the index
        self.index.flush()
        return f"Added data to Qdrant Memory: {data}"  # returns the formatted string indicating input data has been added to the Qdrant memory

    # retrieve a datapoint from Qdrant collection
    def get(self, data):
        return self.get_relevant(data, 1)

    # clear all data points from collection
    def clear(self):
        self.index.clear()
        return "Cleared Qdrant Memory"

    def get_relevant(self, data, num_relevant=5):
        query_embedding = get_ada_embedding(data)
        results = self.index.search([query_embedding], limit=num_relevant)
        return [result.entity.get("source", "") for result in results]

    # retrieve statistics about the Qdrant collection
    def get_stats(self):
        return self.index.get_stats()
