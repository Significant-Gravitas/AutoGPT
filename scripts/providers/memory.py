from config import Singleton
import openai

def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]


def get_text_from_embedding(embedding):
    return openai.Embedding.retrieve(embedding, model="text-embedding-ada-002")["data"][0]["text"]

class Memory(metaclass=Singleton):
    def add(self, data):
        raise NotImplementedError()

    def get(self, data):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()

    def get_relevant(self, data, num_relevant=5):
        raise NotImplementedError()

    def get_stats(self):
        raise NotImplementedError()