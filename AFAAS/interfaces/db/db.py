from __future__ import annotations

import abc
import base64
import json
import uuid
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field, ConfigDict

from AFAAS.configs.schema import Configurable, SystemConfiguration, SystemSettings

if TYPE_CHECKING:
    from AFAAS.interfaces.db.db_table import AbstractTable

from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)


class MemoryAdapterType(Enum):
    SQLLIKE_JSON_FILE = "json_file"
    NOSQL_JSON_FILE = "nosqljson_file"
    DYNAMODB = "dynamodb"
    COSMOSDB = "cosmosdb"
    MONGODB = "mongodb"


class MemoryConfig(SystemConfiguration):
    """
    Configuration class representing the parameters for creating a Memory adapter.

    Attributes:
        db_adapter (MemoryAdapterType): The type of db adapter to use.
        json_file_path (str): The file path for the JSON file when using the JSONFileMemory adapter.
        # Add other parameters for different db adapters as needed.
    """

    db_adapter: MemoryAdapterType = Field(
        MemoryAdapterType.NOSQL_JSON_FILE,
        description="The type of db adapter to use.",
    )
    json_file_path: str = Field(
        str(Path("~/AFAAS/data/pytest").expanduser().resolve()),
        description="The file path for the JSON file when using the JSONFileMemory adapter.",
    )
    # sqllikejson_file_path=str(Path("~/AFAAS/sqlikejsondata/").expanduser().resolve()),
    # cosmos_endpoint=None,
    # cosmos_key=None,
    # cosmos_database_name=None,
    # aws_access_key_id=None,
    # aws_secret_access_key,
    # dynamodb_region_name=None,
    # mongodb_connection_string='connection_string',
    # mongodb_database_name='database_name',
    # mongo_uri=None,
    # mongo_db_name=None


# class Memory.SystemSettings(SystemSettings):
#         configuration: MemoryConfig = MemoryConfig()
#         name: str = "Memory"
#         description: str = "Memory is an abstract db adapter"



class AbstractMemory(Configurable, abc.ABC):
    class SystemSettings(Configurable.SystemSettings):
        configuration: MemoryConfig = MemoryConfig()
        name: str = "Memory"
        description: str = "Memory is an abstract db adapter"

    _instances: dict[AbstractMemory] = {}

    """
    Abstract class representing a db storage system for storing and retrieving data.

    To use a specific db adapter, create a configuration dict specifying the desired
    db_adapter. Currently, "json_file" and "redis" adapters are available.

    Example:
        config = {"db_adapter": "json_file", "json_file_path": "~/AFAAS/data/"}
        db = Memory.get_adapter(config)

    After getting the db adapter, you can connect to it using the `connect` method
    with any required parameters.

    After connecting, you can access individual tables using the `get_table` method,
    passing the desired table name as an argument.

    Example:
        # Assuming we have connected to the db using `db` variable
        agents_table = db.get_table("agents")
        messages_table = db.get_table("messages_history")
        users_table = db.get_table("users")

    Note:
        The `Memory` class is an abstract class, and you should use one of its concrete
        subclasses like `JSONFileMemory` or `RedisMemory` for actual implementations.
    """

    @abc.abstractmethod
    def __init__(
        self,
        settings: AbstractMemory.SystemSettings,
    ):
        AbstractMemory._instances = {}
        super().__init__(settings)

    @classmethod
    async def add_adapter(cls, adapter: AbstractMemory):
        # TODO:v0.1.0 Implement an add adapter method & a more robust multiton with dependency injection
        raise NotImplementedError("add_adapter")
        config_key = base64.b64encode(
            json.dumps(adapter._settings.configuration.dict()).encode()
        ).decode()
        cls._instances[config_key] = adapter

    @classmethod
    def get_adapter(
        cls,
        db_settings: AbstractMemory.SystemSettings = SystemSettings(),
        *args,
        **kwargs,
    ) -> "AbstractMemory":
        """
        Get an instance of a db adapter based on the provided configuration.

        Parameters:
            config (dict): Configuration dict specifying the db_adapter type and
                           any required parameters for that adapter.
            logger (Logger, optional): The logger instance to use for logging messages.
                                       Default: Logger.

        Returns:
            Memory: An instance of the db adapter based on the provided configuration.

        Raises:
            ValueError: If an invalid db_adapter type is provided in the configuration.

        Example:
            config = {"db_adapter": "json_file", "json_file_path": "~/AFAAS/data/"}
            db = Memory.get_adapter(config)
        """
        # FIXME: Move to a dependancy ingestion patern
        adapter_type = db_settings.configuration.db_adapter

        #FIXME: Weakened the key to upgrade to pydantic 2.x
        # config_key = base64.b64encode(
        #     json.dumps(db_settings.configuration.dict()).encode()
        # ).decode()
        config_key = base64.b64encode(adapter_type.__class__.__name__.encode()).decode()

        if config_key in AbstractMemory._instances:
            return AbstractMemory._instances[config_key]

        if adapter_type == MemoryAdapterType.NOSQL_JSON_FILE:
            from AFAAS.core.db.nosql.jsonfile import JSONFileMemory

            LOG.notice(
                "Started using a local JSONFile backend. Help us to implement/test DynamoDB & CosmoDB backends !"
            )
            instance = JSONFileMemory(settings=db_settings)

        elif adapter_type == MemoryAdapterType.SQLLIKE_JSON_FILE:
            raise NotImplementedError("SQLLikeJSONFileMemory")

        elif adapter_type == MemoryAdapterType.DYNAMODB:
            raise NotImplementedError("DynamoDBMemory")

        elif adapter_type == MemoryAdapterType.COSMOSDB:
            raise NotImplementedError("CosmosDBMemory")

        elif adapter_type == MemoryAdapterType.MONGODB:
            raise NotImplementedError("MongoDBMemory")

        else:
            raise ValueError("Invalid db_adapter type")

        AbstractMemory._instances[config_key] = (
            instance  # Store the newly created instance
        )
        return instance

    abc.abstractmethod

    async def get_table(self, table_name: str) -> AbstractTable:
        """
        Get an instance of the table with the specified table_name.

        Parameters:
            table_name (str): The name of the table to retrieve.

        Returns:
            BaseTable: An instance of the table with the specified table_name.

        Raises:
            ValueError: If the provided table_name is not recognized.

        Example:
            # Assuming we have connected to the db using `db` variable
            agents_table = db.get_table("agents")
            messages_table = db.get_table("messages_history")
            users_table = db.get_table("users")
        """
        # FIXME: Move to a dependancy ingestion patern
        if self.__class__ == AbstractMemory:
            raise TypeError(
                "get_table method cannot be called on Memory class directly"
            )

        if table_name == "agents":
            from AFAAS.core.db.table.nosql.agent import AgentsTable

            returnvalue = AgentsTable(db=self)
            return returnvalue

        if table_name == "tasks":
            from AFAAS.core.db.table.nosql.task import TasksTable

            returnvalue = TasksTable(db=self)
            return returnvalue

        elif table_name == "plans":
            from AFAAS.core.db.table.nosql.plan import PlansTable

            returnvalue = PlansTable(db=self)
            return returnvalue

        elif table_name == "message_agent_agent":
            from AFAAS.core.db.table.nosql.message_agent_agent import (
                MessagesAgentAgentTable,
            )

            return MessagesAgentAgentTable(db=self)

        elif table_name == "message_agent_llm":
            from AFAAS.core.db.table.nosql.message_agent_llm import (
                MessagesAgentLLMTable,
            )

            return MessagesAgentLLMTable(db=self)

        elif table_name == "message_agent_user":
            from AFAAS.core.db.table.nosql.message_user_agent import (
                MessagesUserAgentTable,
            )

            return MessagesUserAgentTable(db=self)

        elif table_name == "users_informations":
            from AFAAS.core.db.table.nosql.user import UsersInformationsTable

            return UsersInformationsTable(db=self)

        elif table_name == "artifacts":
            from AFAAS.core.db.table.nosql.artifacts import ArtifactsTable

            return ArtifactsTable(db=self)

        else:
            raise ValueError(f"Unknown table: {table_name}")

    @abc.abstractmethod
    async def connect(self, *args, **kwargs):
        """
        Connect to the db storage system.

        Implement this method to establish a connection to the desired db storage system
        using any required parameters.

        Parameters:
            kwarg: Any required parameters for connecting to the db storage system.

        Example:
            # Example implementation for JSONFileMemory
            async def connect(self, json_file_path):
                # Implementation for connecting to JSON file db storage
                pass
        """

    @abc.abstractmethod
    async def get(self, key: uuid.UUID, table_name: str):
        pass

    @abc.abstractmethod
    async def add(self, key: uuid.UUID, value: dict, table_name: str):
        pass

    @abc.abstractmethod
    async def update(self, key: uuid.UUID, value: dict, table_name: str):
        pass

    @abc.abstractmethod
    async def delete(self, key: uuid.UUID, table_name: str):
        pass

    @abc.abstractmethod
    async def list(self, table_name: str) -> dict:
        pass


class MemoryItem(abc.ABC):
    pass


class MessageHistory(abc.ABC):
    pass
