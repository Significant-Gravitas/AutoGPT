import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import uuid
import os
from datetime import datetime
import random
from memory.base import MemoryProviderSingleton


class ChromaMemory(MemoryProviderSingleton):
    def __init__(self, cfg):
        if cfg.chroma_db_directory is not None:
            self.db = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=cfg.chroma_db_directory))
        elif cfg.chroma_server_host is not None:
            self.db = chromadb.Client(Settings(chroma_db_impl="rest", chroma_server_host=cfg.chroma_server_host, chroma_server_http_port=cfg.chroma_server_port))
        else:
            self.db = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chromadb"))
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction("paraphrase-MiniLM-L6-v2")
        self.collection = self.db.get_or_create_collection(name="autogpt", embedding_function=self.embedding_function)

    def add(self, data):
        _timestamp = (datetime.now() - datetime(1970, 1, 1)).total_seconds()
        if self.collection.add(documents=[data], metadatas=[{"timestamp": _timestamp}], id=[str(uuid.uuid4())]):
            return True
        return False

    def remove(self, data):
        _deleted_id = self.collection.remove(data=[str(uuid.uuid4())])
        _result = f"Deleted {_deleted_id}"
        return _result

    def get(self, data):
        return self.collection.query(query_texts=[data], n_results=1)

    def get_relevant(self, data, num_relevant=5):
        try:
            results = self.collection.query(query_texts=[data], n_results=num_relevant)
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return None
        return results

    def clear(self):
        self.db.reset()
        return "Cleared"

    def get_stats(self):
        return self.collection.count()
