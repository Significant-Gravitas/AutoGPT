import logging
from typing import Any, Literal

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.util.clients import get_database_manager_async_client

logger = logging.getLogger(__name__)


StorageScope = Literal["within_agent", "across_agents"]


def get_storage_key(key: str, scope: StorageScope, graph_id: str) -> str:
    """Generate the storage key based on scope"""
    if scope == "across_agents":
        return f"global#{key}"
    else:
        return f"agent#{graph_id}#{key}"


class PersistInformationBlock(Block):
    """Block for persisting key-value data for the current user with configurable scope"""

    class Input(BlockSchema):
        key: str = SchemaField(description="Key to store the information under")
        value: Any = SchemaField(description="Value to store")
        scope: StorageScope = SchemaField(
            description="Scope of persistence: within_agent (shared across all runs of this agent) or across_agents (shared across all agents for this user)",
            default="within_agent",
        )

    class Output(BlockSchema):
        value: Any = SchemaField(description="Value that was stored")

    def __init__(self):
        super().__init__(
            id="1d055e55-a2b9-4547-8311-907d05b0304d",
            description="Persist key-value information for the current user",
            categories={BlockCategory.DATA},
            input_schema=PersistInformationBlock.Input,
            output_schema=PersistInformationBlock.Output,
            test_input={
                "key": "user_preference",
                "value": {"theme": "dark", "language": "en"},
                "scope": "within_agent",
            },
            test_output=[
                ("value", {"theme": "dark", "language": "en"}),
            ],
            test_mock={
                "_store_data": lambda *args, **kwargs: {
                    "theme": "dark",
                    "language": "en",
                }
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        user_id: str,
        graph_id: str,
        node_exec_id: str,
        **kwargs,
    ) -> BlockOutput:
        # Determine the storage key based on scope
        storage_key = get_storage_key(input_data.key, input_data.scope, graph_id)

        # Store the data
        yield "value", await self._store_data(
            user_id=user_id,
            node_exec_id=node_exec_id,
            key=storage_key,
            data=input_data.value,
        )

    async def _store_data(
        self, user_id: str, node_exec_id: str, key: str, data: Any
    ) -> Any | None:
        return await get_database_manager_async_client().set_execution_kv_data(
            user_id=user_id,
            node_exec_id=node_exec_id,
            key=key,
            data=data,
        )


class RetrieveInformationBlock(Block):
    """Block for retrieving key-value data for the current user with configurable scope"""

    class Input(BlockSchema):
        key: str = SchemaField(description="Key to retrieve the information for")
        scope: StorageScope = SchemaField(
            description="Scope of persistence: within_agent (shared across all runs of this agent) or across_agents (shared across all agents for this user)",
            default="within_agent",
        )
        default_value: Any = SchemaField(
            description="Default value to return if key is not found", default=None
        )

    class Output(BlockSchema):
        value: Any = SchemaField(description="Retrieved value or default value")

    def __init__(self):
        super().__init__(
            id="d8710fc9-6e29-481e-a7d5-165eb16f8471",
            description="Retrieve key-value information for the current user",
            categories={BlockCategory.DATA},
            input_schema=RetrieveInformationBlock.Input,
            output_schema=RetrieveInformationBlock.Output,
            test_input={
                "key": "user_preference",
                "scope": "within_agent",
                "default_value": {"theme": "light", "language": "en"},
            },
            test_output=[
                ("value", {"theme": "light", "language": "en"}),
            ],
            test_mock={"_retrieve_data": lambda *args, **kwargs: None},
            static_output=True,
        )

    async def run(
        self, input_data: Input, *, user_id: str, graph_id: str, **kwargs
    ) -> BlockOutput:
        # Determine the storage key based on scope
        storage_key = get_storage_key(input_data.key, input_data.scope, graph_id)

        # Retrieve the data
        stored_value = await self._retrieve_data(
            user_id=user_id,
            key=storage_key,
        )

        if stored_value is not None:
            yield "value", stored_value
        else:
            yield "value", input_data.default_value

    async def _retrieve_data(self, user_id: str, key: str) -> Any | None:
        return await get_database_manager_async_client().get_execution_kv_data(
            user_id=user_id,
            key=key,
        )
