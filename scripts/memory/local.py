from typing import Any, List, Optional
import os
import uuid
import datetime
import chromadb
from memory.base import MemoryProviderSingleton, get_embedding

class LocalCache(MemoryProviderSingleton):

    # on load, load our database
    def __init__(self, cfg) -> None:
        self.persistence = cfg.memory_directory
        if os.path.exists(self.persistence):
            self.chromaClient = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet", # duckdb+parquet = persisted, duckdb = in-memory
                persist_directory=self.persistence
            ))
        else:
            # in memory
            print(f"Warning: The directory '{self.persistence}' does not exist. Chroma memory would not be saved to a file.")
            self.chromaClient = chromadb.Client()
        self.chromaCollection = self.chromaClient.create_collection(name="autoGPT_collection")
        # we will key off of cfg.openai_embeddings_model to determine if using sentence transformers or openai embeddings
        self.useOpenAIEmbeddings = True if (cfg.openai_embeddings_model) else False

    def add(self, text: str):
        current_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
        metadata = {"time_added": current_time}
        if self.useOpenAIEmbeddings:
            embeddings = get_embedding(text)
            self.chromaCollection.add(
                embeddings=[embeddings],
                ids=[str(uuid.uuid4())],
                metadatas=[metadata]
            )
        else:
            self.chromaCollection.add(
                documents=[text],
                ids=[str(uuid.uuid4())],
                metadatas=[metadata]
            )
        return text

    def clear(self) -> str:
        """
        Resets the Chroma database.

        Returns: A message indicating that the db has been cleared.
        """
        
        chroma_client = self.chromaClient
        chroma_client.reset()
        self.chromaCollection = chroma_client.create_collection(name="autoGPT_collection")
        return "Obliviated"

    def get(self, data: str) -> Optional[List[Any]]:
        """
        Gets the data from the memory that is most relevant to the given data.

        Args:
            data: The data to compare to.

        Returns: The most relevant data.
        """
        results = None
        if self.useOpenAIEmbeddings:
            embeddings = get_embedding(data)
            results = self.collection.query(
                query_embeddings=[data],
                n_results=1
            )
        else:
            results = self.collection.query(
                query_texts=[data],
                n_results=1
            )
        return results

    def get_relevant(self, text: str, k: int) -> List[Any]:
        results = None
        if self.useOpenAIEmbeddings:
            embeddings = get_embedding(data)
            results = self.collection.query(
                query_embeddings=[data],
                n_results=k
            )
        else:
            results = self.collection.query(
                query_texts=[data],
                n_results=k
            )
        return results

    def get_stats(self):
        """
        Returns: The number of items that have been added to the Chroma collection
        """
        return self.chromaCollection.count()
