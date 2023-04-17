import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
from autogpt.memory.base import MemoryProviderSingleton, get_ada_embedding


class QdrantMemory(MemoryProviderSingleton):
    def __init__(self, cfg):
        self.qdrant_host = cfg.qdrant_host
        self.qdrant_port = cfg.qdrant_port
        self.dimension = 1536
        self.metric = rest.Distance["COSINE"]
        self.index_name = cfg.memory_index
        self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        self.vec_num = uuid.uuid4().hex

        # Check connection
        if not self.ping():
            raise ConnectionError(
                f"Unable to connect to Qdrant server at {self.qdrant_host}:{self.qdrant_port}"
            )

        # Create index if not exists
        try:
            self.client.get_collection(self.index_name)
        except UnexpectedResponse:
            self.client.recreate_collection(
                self.index_name,
                vectors_config=rest.VectorParams(
                    size=self.dimension,
                    distance=self.metric
                )
            )

    def ping(self):
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            print(f"Error while connecting to Qdrant server: {e}")
            return False

    def add(self, data):
        vector = get_ada_embedding(data)
        point = PointStruct(
            id=self.vec_num,
            vector=vector,
            payload={"raw_text": data},
        )
        resp = self.client.upsert(
            collection_name=self.index_name,
            points=[point],
            wait=True
        )
        _text = f"Inserting data into memory at index: {self.vec_num}:\n data: {data}"
        self.vec_num = uuid.uuid4().hex
        return _text

    def get(self, data):
        return self.get_relevant(data, 1)

    def clear(self):
        self.client.delete_collection(self.index_name);
        self.client.recreate_collection(
            self.index_name,
            vectors_config=rest.VectorParams(
                size=self.dimension,
                distance=self.metric
            )
        )
        self.vec_num = 0
        return "Obliviated"

    def get_relevant(self, data, num_relevant=5):
        query_embedding = get_ada_embedding(data)
        results = self.client.search(
            collection_name=self.index_name,
            query_vector=query_embedding,
            top=num_relevant,
            payload=True,
        )
        sorted_results = sorted(results, key=lambda x: -x.score)[:num_relevant]

        return [item.payload["raw_text"] for item in sorted_results]

    def get_stats(self):
        collection_info = self.client.get_collection(collection_name=self.index_name)
        return {
            "optimizer_status": collection_info.optimizer_status,
            "vectors_count": collection_info.vectors_count,
            "indexed_vectors_count": collection_info.indexed_vectors_count,
            "points_count": collection_info.points_count,
            "segments_count": collection_info.segments_count,
        }
