from typing import Any, List, Optional, Tuple
import os
import uuid
import datetime
import chromadb
import logging
from autogpt.memory.base import MemoryProviderSingleton
from autogpt.llm_utils import create_embedding
from chromadb.errors import NoIndexException


class LocalCache(MemoryProviderSingleton):
    """A class that stores the memory in a local file"""

    def __init__(self, cfg) -> None:
        """Initialize a class instance

        Args:
            cfg: Config object

        Returns:
            None
        """

        # switch chroma's logging defaults to error only
        logging.getLogger('chromadb').setLevel(logging.ERROR)
        

        self.chromaClient = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet", # this makes it persisted, comment out to be purely in-memory
            persist_directory="localCache"
        ))
        self.chromaCollection = self.chromaClient.create_collection(name="autogpt")
        # we will key off of cfg.openai_embeddings_model to determine if using sentence transformers or openai embeddings
        self.useOpenAIEmbeddings = True if (cfg.openai_embeddings_model) else False

    def add(self, text: str):
        """
        Add text to our list of texts, add embedding as row to our
            embeddings-matrix

        Args:
            text: str

        Returns: None
        """
        if "Command Error:" in text:
            return ""

        current_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
        metadata = {"time_added": current_time}
        if self.useOpenAIEmbeddings:
            embeddings = create_embedding(text)
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
        self.chromaCollection = chroma_client.create_collection(name="autogpt")
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
            embeddings = create_embedding(data)
            results = self.chromaCollection.query(
                query_embeddings=[data],
                n_results=1
            )
        else:
            results = self.chromaCollection.query(
                query_texts=[data],
                n_results=1
            )
        return results

    def get_relevant(self, text: str, k: int) -> List[Any]:
        results = None
        try: 
            if self.useOpenAIEmbeddings:
                embeddings = create_embedding(text)
                results = self.chromaCollection.query(
                    query_embeddings=[text],
                    n_results=min(k, self.chromaCollection.count())
                )
            else:
                results = self.chromaCollection.query(
                    query_texts=[text],
                    n_results=min(k, self.chromaCollection.count())
                )
        except NoIndexException:
            # print("No index found - suppressed because this is a common issue for first-run users")
            pass
        return results

    def get_stats(self) -> Tuple[int, Tuple[int, ...]]:
        """
        Returns: The number of items that have been added to the Chroma collection
        """
        return self.chromaCollection.count()
