from config import config,Singleton
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import openai
import uuid
import os
from datetime import datetime
import random

class ChromaMemory(metaclass=Singleton):
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

    def add(self, text):
        _timestamp = (datetime.now() - datetime(1970, 1, 1)).total_seconds()
        if self.collection.add(documents=[text], metadatas=[{"timestamp": _timestamp}], id=[str(uuid.uuid4())]):
            return True
        return False

    def remove(self, id):
        _deleted_id = self.collection.remove(id=[str(uuid.uuid4())])
        _result = f"Deleted {_deleted_id}"
        return _result

    def get(self, text, count=5):
        return self.collection.query(query_texts=[text], n_results=count)
    
    def serendipity_utility(self, text, breadth=50, depth=5):
        # In order to prevent the same cluster of results from being returned from the agent's long-term memory within the immediate context of the conversation, 
        # Here is an algorithm to "spike" the context with a random, yet, related result farther back in time. 
        # This algorithm samples 50 related objects (or another value as denoted by "breadth") from the long-term memory, sorts them, then randomly samples 1 item from the oldest 5 (or other sample size as denoted by "depth").
        _results = self.collection.query(query_texts=[text], n_results=breadth)
        _curr_time = (datetime.now() - datetime(1970, 1, 1)).total_seconds()
        _ordered_results = []
        for i in range(len(_results.ids)):
            _ordered_results.append({"text": _results.texts[i], "age": (_curr_time - _results.metadatas[i]["timestamp"])})
        _ordered_results.sort(key=lambda x: x["age"])
        _serendipity_subset = _ordered_results[:depth]
        _serendipity_result = _serendipity_subset[random.randint(0, len(_serendipity_subset)-1)].text
        return _serendipity_result

    

    def clear(self):
        self.collection.clear()
        return "Cleared"