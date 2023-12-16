from __future__ import annotations


from typing import TYPE_CHECKING, List

from pymongo import MongoClient

if TYPE_CHECKING:
    from AFAAS.interfaces.db import AbstractMemory

from AFAAS.interfaces.db_nosql import NoSQLMemory

from AFAAS.core.lib.sdk.logger import AFAASLogger
LOG = AFAASLogger(name=__name__)
class MongoDBMemory(NoSQLMemory):
    """
    DO NOT USE : TEMPLATE UNDER DEVELOPMENT, WOULD HAPPILY TAKE HELP :-)

    Args:
        Memory (_type_): _description_
    """

    def __init__(
        self,
        settings: AbstractMemory.SystemSettings,
    ):
        super().__init__(settings)
        self._client = None
        self._db = None

    def connect(self, mongo_uri=None, mongo_db_name=None):
        # Extracting values from arguments
        uri = mongo_uri | self.mongo_uri
        mongo_db_name = mongo_db_name | self.mongo_db_name

        # Establishing connection to MongoDB
        self._client = MongoClient(uri)
        self._db = self._client[mongo_db_name]

        try:
            # Checking if the connection is successful
            self._db.list_collection_names()
        except Exception as e:
            # Handling connection error
            LOG.error(f"Unable to connect to MongoDB: {e}")
            raise e
        else:
            # Connection successful
            LOG.info("Successfully connected to MongoDB.")

    def get(self, key: dict, table_name: str):
        collection = self._db[table_name]
        return collection.find_one(key)

    def add(self, key: dict, value: dict, table_name: str):
        collection = self._db[table_name]
        value.update(key)
        collection.insert_one(value)

    def update(self, key: dict, value: dict, table_name: str):
        collection = self._db[table_name]
        updated_result = collection.update_one(key, {"$set": value})
        if updated_result.matched_count == 0:
            raise KeyError(f"No such key '{key}' in table {table_name}")

    def delete(self, key: dict, table_name: str):
        collection = self._db[table_name]
        delete_result = collection.delete_one(key)
        if delete_result.deleted_count == 0:
            raise KeyError(f"No such key '{key}' in table {table_name}")

    def list(self, table_name: str) -> list[dict]:
        collection = self._db[table_name]
        return list(collection.find({}))
