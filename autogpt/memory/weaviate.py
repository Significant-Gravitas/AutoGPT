from autogpt.config import Config
from autogpt.memory.base import MemoryProviderSingleton, get_ada_embedding
import uuid
import weaviate
from weaviate import Client
from weaviate.embedded import EmbeddedOptions
from weaviate.util import generate_uuid5


def default_schema(weaviate_index):
    return {
        "class": weaviate_index,
        "properties": [
            {
                "name": "raw_text",
                "dataType": ["text"],
                "description": "original text for the embedding"
            }
        ],
    }


class WeaviateMemory(MemoryProviderSingleton):
    def __init__(self, cfg):
        auth_credentials = self._build_auth_credentials(cfg)

        url = f'{cfg.weaviate_protocol}://{cfg.weaviate_host}:{cfg.weaviate_port}'

        if cfg.use_weaviate_embedded:
            self.client = Client(embedded_options=EmbeddedOptions(
                hostname=cfg.weaviate_host,
                port=int(cfg.weaviate_port),
                persistence_data_path=cfg.weaviate_embedded_path
            ))

            print(f"Weaviate Embedded running on: {url} with persistence path: {cfg.weaviate_embedded_path}")
        else:
            self.client = Client(url, auth_client_secret=auth_credentials)

        self.index = cfg.memory_index
        self._create_schema()

    def _create_schema(self):
        schema = default_schema(self.index)
        if not self.client.schema.contains(schema):
            self.client.schema.create_class(schema)

    def _build_auth_credentials(self, cfg):
        if cfg.weaviate_username and cfg.weaviate_password:
            return weaviate.AuthClientPassword(cfg.weaviate_username, cfg.weaviate_password)
        if cfg.weaviate_api_key:
            return weaviate.AuthApiKey(api_key=cfg.weaviate_api_key)
        else:
            return None

    def add(self, data):
        vector = get_ada_embedding(data)

        doc_uuid = generate_uuid5(data, self.index)
        data_object = {
            'raw_text': data
        }

        with self.client.batch as batch:
            batch.add_data_object(
                uuid=doc_uuid,
                data_object=data_object,
                class_name=self.index,
                vector=vector
            )

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
            results = self.client.query.get(self.index, ['raw_text']) \
                          .with_near_vector({'vector': query_embedding, 'certainty': 0.7}) \
                          .with_limit(num_relevant)  \
                          .do()

            if len(results['data']['Get'][self.index]) > 0:
                return [str(item['raw_text']) for item in results['data']['Get'][self.index]]
            else:
                return []

        except Exception as err:
            print(f'Unexpected error {err=}, {type(err)=}')
            return []

    def get_stats(self):
        result = self.client.query.aggregate(self.index) \
                     .with_meta_count() \
                     .do()
        class_data = result['data']['Aggregate'][self.index]

        return class_data[0]['meta'] if class_data else {}
