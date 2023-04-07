from config import Config
from providers.memory import Memory, get_ada_embedding
from weaviate import Client
import weaviate
import uuid

from weaviate.util import generate_uuid5

cfg = Config()

SCHEMA = {
    "class": cfg.weaviate_index,
    "properties": [
        {
            "name": "raw_text",
            "dataType": ["text"],
            "description": "original text for the embedding"
        }
    ],
}

class WeaviateMemory(Memory):

    def __init__(self):
        auth_credentials = self._build_auth_credentials()

        url = f'{cfg.weaviate_host}:{cfg.weaviate_port}'

        self.client = Client(url, auth_client_secret=auth_credentials)

        self._create_schema()

    def _create_schema(self):
        if not self.client.schema.contains(SCHEMA):
            self.client.schema.create_class(SCHEMA)

    @staticmethod
    def _build_auth_credentials():
        if cfg.weaviate_username and cfg.weaviate_password:
            return weaviate_auth.AuthClientPassword(cfg.weaviate_username, cfg.weaviate_password)
        else:
            return None

    def add(self, data):
        vector = get_ada_embedding(data)

        doc_uuid = generate_uuid5(data, cfg.weaviate_index)
        data_object = {
            'class': cfg.weaviate_index,
            'raw_text': data
        }

        with self.client.batch as batch:
            batch.add_data_object(
                uuid=doc_uuid,
                data_object=data_object,
                class_name=cfg.weaviate_index,
                vector=vector
            )

            batch.flush()

        return f"Inserting data into memory at uuid: {doc_uuid}:\n data: {data}"


    def get(self, data):
        return self.get_relevant(data, 1)


    def clear(self):
        self.client.schema.delete_all()

        # weaviate does not yet have a neat way to just remove the items in an index
        # without removing the entire schema, therefore we need to re-create it 
        # after a call to delete_all
        self._create_schema()

        return 'Obliterated'

    def get_relevant(self, data, num_relevant=5):
        query_embedding = get_ada_embedding(data)
        try:
            results = self.client.query.get(cfg.weaviate_index, ['raw_text']) \
                          .with_near_vector({'vector': query_embedding, 'certainty': 0.7}) \
                          .with_limit(num_relevant)  \
                          .do()

            print(results)
             
            if len(results['data']['Get'][cfg.weaviate_index]) > 0:
                return [str(item['raw_text']) for item in results['data']['Get'][cfg.weaviate_index]]
            else:
                return []

        except Exception as err:
            print(f'Unexpected error {err=}, {type(err)=}')
            return []

    def get_stats(self):
        return self.client.index_stats.get(cfg.weaviate_index)