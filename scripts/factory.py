from providers.pinecone import PineconeMemory
from providers.weaviate import WeaviateMemory

class MemoryFactory:
    @staticmethod
    def get_memory(mem_type):
        if mem_type == 'pinecone':
            return PineconeMemory()

        if mem_type == 'weaviate':
            return WeaviateMemory()

        raise ValueError('Unknown memory provider')