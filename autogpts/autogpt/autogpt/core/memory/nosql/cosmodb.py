from __future__ import annotations

from logging import Logger
from typing import TYPE_CHECKING, List

from azure.cosmos import CosmosClient

if TYPE_CHECKING:
    from autogpts.autogpt.autogpt.core.memory.base import AbstractMemory

from autogpts.autogpt.autogpt.core.memory.nosqlmemory import NoSQLMemory


class CosmosDBMemory(NoSQLMemory):
    """
    DO NOT USE : TEMPLATE UNDER DEVELOPMENT, WOULD HAPPILY TAKE HELP :-)

    Args:
        Memory (_type_): _description_
    """

    def __init__(
        self,
        settings: AbstractMemory.SystemSettings,
        logger: Logger,
    ):
        super().__init__(settings, logger)
        self._client = None
        self._database = None
        self._logger = logger

    def connect(self, cosmos_endpoint=None, cosmos_key=None, cosmos_database_name=None):
        endpoint = cosmos_endpoint | self.cosmos_endpoint
        db_name = cosmos_database_name | self.cosmos_database_name
        key = cosmos_key | self.cosmos_key
        self._client = CosmosClient(endpoint, key)
        self._database = self._client.get_database_client(db_name)
        try:
            self._database.read()
        except Exception as e:
            self._logger.error(f"Unable to connect to Cosmos DB: {e}")
            raise e
        else:
            self._logger.info("Successfully connected to Cosmos DB.")

    def get(self, key: dict, table_name: str):
        container = self._database.get_container_client(table_name)
        query = f"SELECT * FROM c WHERE c.id = '{key['primary_key']}'"
        results = list(
            container.query_items(query=query, enable_cross_partition_query=True)
        )
        return results[0] if results else None

    def add(self, key: dict, value: dict, table_name: str):
        container = self._database.get_container_client(table_name)
        value.update(key)
        container.upsert_item(value)

    def update(self, key: dict, value: dict, table_name: str):
        container = self._database.get_container_client(table_name)
        query = f"SELECT * FROM c WHERE c.id = '{key['primary_key']}'"
        results = list(
            container.query_items(query=query, enable_cross_partition_query=True)
        )
        if not results:
            raise KeyError(f"No such key '{key}' in table {table_name}")
        item = results[0]
        item.update(value)
        container.replace_item(item, item)

    def delete(self, key: dict, table_name: str):
        container = self._database.get_container_client(table_name)
        container.delete_item(item=key["primary_key"], partition_key=key["primary_key"])

    def list(self, table_name: str) -> list[dict]:
        container = self._database.get_container_client(table_name)
        return list(container.read_all_items())
