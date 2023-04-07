from config import Config, Singleton
import pinecone
import openai

cfg = Config()


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]


def get_text_from_embedding(embedding):
    return openai.Embedding.retrieve(embedding, model="text-embedding-ada-002")["data"][0]["text"]


class PineconeMemory(metaclass=Singleton):
    
    # The default namespace in Pinecone. The assumption here is that it will be overridden 
    _pinecone_namespace = ""
    
    def __init__(self, ns=_pinecone_namespace):
        pinecone_api_key = cfg.pinecone_api_key
        pinecone_region = cfg.pinecone_region
        self._pinecone_namespace = ns
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_region)
        dimension = 1536
        metric = "cosine"
        pod_type = "p1"
        table_name = "auto-gpt"

        if table_name not in pinecone.list_indexes():
            pinecone.create_index(table_name, dimension=dimension, metric=metric, pod_type=pod_type)
        self.index = pinecone.Index(table_name)

        # figure out what vector number to start with by pulling the total vector count
        index_stats = self.index.describe_index_stats()
        self.vec_num = index_stats._data_store.get('total_vector_count')

    def add(self, data, ns=_pinecone_namespace):
        vector = get_ada_embedding(data)
        # no metadata here. We may wish to change that long term.
        resp = self.index.upsert([(str(self.vec_num), vector, {"raw_text": data})], namespace=ns)
        _text = f"Inserting data into memory at index: {self.vec_num}:\n data: {data}"
        self.vec_num += 1
        return _text

    def get(self, data, ns=_pinecone_namespace):
        return self.get_relevant(data, ns, 1)

    def clear(self, ns=_pinecone_namespace):
        self.index.delete(delete_all=True, namespace=ns)
        return "Obliviated"

    def get_relevant(self, data, ns=_pinecone_namespace, num_relevant=5):
        """
        Returns all the data in the memory that is relevant to the given data.
        :param data: The data to compare to.
        :param num_relevant: The number of relevant data to return. Defaults to 5
        """
        query_embedding = get_ada_embedding(data)
        results = self.index.query(query_embedding, top_k=num_relevant, include_metadata=True, namespace=ns)
        sorted_results = sorted(results.matches, key=lambda x: x.score)
        return [str(item['metadata']["raw_text"]) for item in sorted_results]

    def get_stats(self):
        return self.index.describe_index_stats()
