from typing import Any, Literal, Optional, Union

from mem0 import MemoryClient
from pydantic import BaseModel, SecretStr

from backend.blocks._base import Block, BlockOutput, BlockSchemaInput, BlockSchemaOutput
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName

TEST_CREDENTIALS = APIKeyCredentials(
    id="8cc8b2c5-d3e4-4b1c-84ad-e1e9fe2a0122",
    provider="mem0",
    api_key=SecretStr("mock-mem0-api-key"),
    title="Mock Mem0 API key",
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


class Mem0Base:
    """Base class with shared utilities for Mem0 blocks"""

    @staticmethod
    def _get_client(credentials: APIKeyCredentials) -> MemoryClient:
        """Get initialized Mem0 client"""
        return MemoryClient(api_key=credentials.api_key.get_secret_value())


Filter = dict[str, list[dict[str, str | dict[str, list[str]]]]]


class Conversation(BaseModel):
    discriminator: Literal["conversation"]
    messages: list[dict[str, str]]


class Content(BaseModel):
    discriminator: Literal["content"]
    content: str


class AddMemoryBlock(Block, Mem0Base):
    """Block for adding memories to Mem0

    Always limited by user_id and optional graph_id and graph_exec_id"""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.MEM0], Literal["api_key"]
        ] = CredentialsField(description="Mem0 API key credentials")
        content: Union[Content, Conversation] = SchemaField(
            discriminator="discriminator",
            description="Content to add - either a string or list of message objects as output from an AI block",
            default=Content(discriminator="content", content="I'm a vegetarian"),
        )
        metadata: dict[str, Any] = SchemaField(
            description="Optional metadata for the memory", default_factory=dict
        )
        limit_memory_to_run: bool = SchemaField(
            description="Limit the memory to the run", default=False
        )
        limit_memory_to_agent: bool = SchemaField(
            description="Limit the memory to the agent", default=True
        )

    class Output(BlockSchemaOutput):
        action: str = SchemaField(description="Action of the operation")
        memory: str = SchemaField(description="Memory created")
        results: list[dict[str, str]] = SchemaField(
            description="List of all results from the operation"
        )

    def __init__(self):
        super().__init__(
            id="dce97578-86be-45a4-ae50-f6de33fc935a",
            description="Add new memories to Mem0 with user segmentation",
            input_schema=AddMemoryBlock.Input,
            output_schema=AddMemoryBlock.Output,
            test_input=[
                {
                    "content": {
                        "discriminator": "conversation",
                        "messages": [{"role": "user", "content": "I'm a vegetarian"}],
                    },
                    "metadata": {"food": "vegetarian"},
                    "credentials": TEST_CREDENTIALS_INPUT,
                },
                {
                    "content": {
                        "discriminator": "content",
                        "content": "I am a vegetarian",
                    },
                    "metadata": {"food": "vegetarian"},
                    "credentials": TEST_CREDENTIALS_INPUT,
                },
            ],
            test_output=[
                ("results", [{"event": "CREATED", "memory": "test memory"}]),
                ("action", "CREATED"),
                ("memory", "test memory"),
                ("results", [{"event": "CREATED", "memory": "test memory"}]),
                ("action", "CREATED"),
                ("memory", "test memory"),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={"_get_client": lambda credentials: MockMemoryClient()},
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        user_id: str,
        graph_id: str,
        graph_exec_id: str,
        **kwargs,
    ) -> BlockOutput:
        try:
            client = self._get_client(credentials)

            if isinstance(input_data.content, Conversation):
                messages = input_data.content.messages
            elif isinstance(input_data.content, Content):
                messages = [{"role": "user", "content": input_data.content.content}]
            else:
                messages = [{"role": "user", "content": str(input_data.content)}]

            params = {
                "user_id": user_id,
                "output_format": "v1.1",
                "metadata": input_data.metadata,
            }

            if input_data.limit_memory_to_run:
                params["run_id"] = graph_exec_id
            if input_data.limit_memory_to_agent:
                params["agent_id"] = graph_id

            # Use the client to add memory
            result = client.add(
                messages,
                **params,
            )

            results = result.get("results", [])
            yield "results", results

            if len(results) > 0:
                for result in results:
                    yield "action", result["event"]
                    yield "memory", result["memory"]
            else:
                yield "action", "NO_CHANGE"

        except Exception as e:
            yield "error", str(e)


class SearchMemoryBlock(Block, Mem0Base):
    """Block for searching memories in Mem0"""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.MEM0], Literal["api_key"]
        ] = CredentialsField(description="Mem0 API key credentials")
        query: str = SchemaField(
            description="Search query",
            advanced=False,
        )
        trigger: bool = SchemaField(
            description="An unused field that is used to (re-)trigger the block when you have no other inputs",
            default=False,
            advanced=False,
        )
        categories_filter: list[str] = SchemaField(
            description="Categories to filter by",
            default_factory=list,
            advanced=True,
        )
        metadata_filter: Optional[dict[str, Any]] = SchemaField(
            description="Optional metadata filters to apply",
            default=None,
        )
        limit_memory_to_run: bool = SchemaField(
            description="Limit the memory to the run", default=False
        )
        limit_memory_to_agent: bool = SchemaField(
            description="Limit the memory to the agent", default=True
        )

    class Output(BlockSchemaOutput):
        memories: Any = SchemaField(description="List of matching memories")

    def __init__(self):
        super().__init__(
            id="bd7c84e3-e073-4b75-810c-600886ec8a5b",
            description="Search memories in Mem0 by user",
            input_schema=SearchMemoryBlock.Input,
            output_schema=SearchMemoryBlock.Output,
            test_input={
                "query": "vegetarian preferences",
                "credentials": TEST_CREDENTIALS_INPUT,
                "top_k": 10,
                "rerank": True,
            },
            test_output=[
                ("memories", [{"id": "test-memory", "content": "test content"}])
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={"_get_client": lambda credentials: MockMemoryClient()},
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        user_id: str,
        graph_id: str,
        graph_exec_id: str,
        **kwargs,
    ) -> BlockOutput:
        try:
            client = self._get_client(credentials)

            filters: Filter = {
                # This works with only one filter, so we can allow others to add on later
                "AND": [
                    {"user_id": user_id},
                ]
            }
            if input_data.categories_filter:
                filters["AND"].append(
                    {"categories": {"contains": input_data.categories_filter}}
                )
            if input_data.limit_memory_to_run:
                filters["AND"].append({"run_id": graph_exec_id})
            if input_data.limit_memory_to_agent:
                filters["AND"].append({"agent_id": graph_id})
            if input_data.metadata_filter:
                filters["AND"].append({"metadata": input_data.metadata_filter})

            result: list[dict[str, Any]] = client.search(
                input_data.query, version="v2", filters=filters
            )
            yield "memories", result

        except Exception as e:
            yield "error", str(e)


class GetAllMemoriesBlock(Block, Mem0Base):
    """Block for retrieving all memories from Mem0"""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.MEM0], Literal["api_key"]
        ] = CredentialsField(description="Mem0 API key credentials")
        trigger: bool = SchemaField(
            description="An unused field that is used to trigger the block when you have no other inputs",
            default=False,
            advanced=False,
        )
        categories: Optional[list[str]] = SchemaField(
            description="Filter by categories", default=None
        )
        metadata_filter: Optional[dict[str, Any]] = SchemaField(
            description="Optional metadata filters to apply",
            default=None,
        )
        limit_memory_to_run: bool = SchemaField(
            description="Limit the memory to the run", default=False
        )
        limit_memory_to_agent: bool = SchemaField(
            description="Limit the memory to the agent", default=True
        )

    class Output(BlockSchemaOutput):
        memories: Any = SchemaField(description="List of memories")

    def __init__(self):
        super().__init__(
            id="45aee5bf-4767-45d1-a28b-e01c5aae9fc1",
            description="Retrieve all memories from Mem0 with optional conversation filtering",
            input_schema=GetAllMemoriesBlock.Input,
            output_schema=GetAllMemoriesBlock.Output,
            test_input={
                "metadata_filter": {"type": "test"},
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                ("memories", [{"id": "test-memory", "content": "test content"}]),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={"_get_client": lambda credentials: MockMemoryClient()},
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        user_id: str,
        graph_id: str,
        graph_exec_id: str,
        **kwargs,
    ) -> BlockOutput:
        try:
            client = self._get_client(credentials)

            filters: Filter = {
                "AND": [
                    {"user_id": user_id},
                ]
            }
            if input_data.limit_memory_to_run:
                filters["AND"].append({"run_id": graph_exec_id})
            if input_data.limit_memory_to_agent:
                filters["AND"].append({"agent_id": graph_id})
            if input_data.categories:
                filters["AND"].append(
                    {"categories": {"contains": input_data.categories}}
                )
            if input_data.metadata_filter:
                filters["AND"].append({"metadata": input_data.metadata_filter})

            memories: list[dict[str, Any]] = client.get_all(
                filters=filters,
                version="v2",
            )

            yield "memories", memories

        except Exception as e:
            yield "error", str(e)


class GetLatestMemoryBlock(Block, Mem0Base):
    """Block for retrieving the latest memory from Mem0"""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.MEM0], Literal["api_key"]
        ] = CredentialsField(description="Mem0 API key credentials")
        trigger: bool = SchemaField(
            description="An unused field that is used to trigger the block when you have no other inputs",
            default=False,
            advanced=False,
        )
        categories: Optional[list[str]] = SchemaField(
            description="Filter by categories", default=None
        )
        conversation_id: Optional[str] = SchemaField(
            description="Optional conversation ID to retrieve the latest memory from (uses run_id)",
            default=None,
        )
        metadata_filter: Optional[dict[str, Any]] = SchemaField(
            description="Optional metadata filters to apply",
            default=None,
        )
        limit_memory_to_run: bool = SchemaField(
            description="Limit the memory to the run", default=False
        )
        limit_memory_to_agent: bool = SchemaField(
            description="Limit the memory to the agent", default=True
        )

    class Output(BlockSchemaOutput):
        memory: Optional[dict[str, Any]] = SchemaField(
            description="Latest memory if found"
        )
        found: bool = SchemaField(description="Whether a memory was found")

    def __init__(self):
        super().__init__(
            id="0f9d81b5-a145-4c23-b87f-01d6bf37b677",
            description="Retrieve the latest memory from Mem0 with optional key filtering",
            input_schema=GetLatestMemoryBlock.Input,
            output_schema=GetLatestMemoryBlock.Output,
            test_input={
                "metadata_filter": {"type": "test"},
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                ("memory", {"id": "test-memory", "content": "test content"}),
                ("found", True),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={"_get_client": lambda credentials: MockMemoryClient()},
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        user_id: str,
        graph_id: str,
        graph_exec_id: str,
        **kwargs,
    ) -> BlockOutput:
        try:
            client = self._get_client(credentials)

            filters: Filter = {
                "AND": [
                    {"user_id": user_id},
                ]
            }
            if input_data.limit_memory_to_run:
                filters["AND"].append({"run_id": graph_exec_id})
            if input_data.limit_memory_to_agent:
                filters["AND"].append({"agent_id": graph_id})
            if input_data.categories:
                filters["AND"].append(
                    {"categories": {"contains": input_data.categories}}
                )
            if input_data.metadata_filter:
                filters["AND"].append({"metadata": input_data.metadata_filter})

            memories: list[dict[str, Any]] = client.get_all(
                filters=filters,
                version="v2",
            )

            if memories:
                # Return the latest memory (first in the list as they're sorted by recency)
                latest_memory = memories[0]
                yield "memory", latest_memory
                yield "found", True
            else:
                yield "memory", None
                yield "found", False

        except Exception as e:
            yield "error", str(e)


# Mock client for testing
class MockMemoryClient:
    """Mock Mem0 client for testing"""

    def add(self, *args, **kwargs):
        return {"results": [{"event": "CREATED", "memory": "test memory"}]}

    def search(self, *args, **kwargs) -> list[dict[str, Any]]:
        return [{"id": "test-memory", "content": "test content"}]

    def get_all(self, *args, **kwargs) -> list[dict[str, str]]:
        return [{"id": "test-memory", "content": "test content"}]
