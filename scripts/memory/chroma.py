import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import uuid
import os
from datetime import datetime
import random
from memory.base import MemoryProviderSingleton

class ChromaMemory(MemoryProviderSingleton):
    def __init__(self):
        if os.getenv("CHROMA-DB-DIRECTORY") is not None:
            self.db = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=os.getenv("CHROMA-DB-DIRECTORY")))
        if os.getenv("CHROMA-SERVER-HOST") is not None:
            if os.getenv("CHROMA-SERVER-PORT") is not None:
                self.db = chromadb.Client(Settings(chroma_db_impl="rest", chroma_server_host=os.getenv("CHROMA-SERVER-HOST"), chroma_server_http_port=os.getenv("CHROMA-SERVER-PORT")))
            else:
                self.db = chromadb.Client(Settings(chroma_db_impl="rest", chroma_server_host=os.getenv("CHROMA-SERVER-HOST"), chroma_server_http_port=8000))
        else:
            self.db = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet"))
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

    def get_relevant(self, data, num_relevant=5):
        return self.collection.query(query_texts=[data], n_results=num_relevant)

    def clear(self):
        self.collection.clear()
        return "Cleared"
    
    def get_stats(self):
        return self.collection.count()