from typing import Any, Literal, Optional, Union

from mem0 import MemoryClient
from pydantic import BaseModel, SecretStr

from backend.data.block import Block, BlockOutput, BlockSchema
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName

TEST_CREDENTIALS = APIKeyCredentials(
    id="ed55ac19-356e-4243-a6cb-bc599e9b716f",
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

    class Input(BlockSchema):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.MEM0], Literal["api_key"]
        ] = CredentialsField(description="Mem0 API key credentials")
        content: Union[Content, Conversation] = SchemaField(
            discriminator="discriminator",
            description="Content to add - either a string or list of message objects as output from an AI block",
            default=Content(discriminator="content", content="I'm a vegetarian"),
        )
        metadata: dict[str, Any] = SchemaField(
            description="Optional metadata for the memory", default={}
        )

        limit_memory_to_run: bool = SchemaField(
            description="Limit the memory to the run", default=False
        )
        limit_memory_to_agent: bool = SchemaField(
            description="Limit the memory to the agent", default=False
        )

    class Output(BlockSchema):
        action: str = SchemaField(description="Action of the operation")
        memory: str = SchemaField(description="Memory created")
        error: str = SchemaField(description="Error message if operation fails")

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
            test_output=[("action", "NO_CHANGE"), ("action", "NO_CHANGE")],
            test_credentials=TEST_CREDENTIALS,
            test_mock={"_get_client": lambda credentials: MockMemoryClient()},
        )

    def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        user_id: str,
        graph_id: str,
        graph_exec_id: str,
        **kwargs
    ) -> BlockOutput:
        try:
            client = self._get_client(credentials)

            if isinstance(input_data.content, Conversation):
                messages = input_data.content.messages
            else:
                messages = [{"role": "user", "content": input_data.content}]

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

            if len(result.get("results", [])) > 0:
                for result in result.get("results", []):
                    yield "action", result["event"]
                    yield "memory", result["memory"]
            else:
                yield "action", "NO_CHANGE"

        except Exception as e:
            yield "error", str(object=e)


class SearchMemoryBlock(Block, Mem0Base):
    """Block for searching memories in Mem0"""

    class Input(BlockSchema):
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
            default=[],
            advanced=True,
        )
        limit_memory_to_run: bool = SchemaField(
            description="Limit the memory to the run", default=False
        )
        limit_memory_to_agent: bool = SchemaField(
            description="Limit the memory to the agent", default=True
        )

    class Output(BlockSchema):
        memories: Any = SchemaField(description="List of matching memories")
        error: str = SchemaField(description="Error message if operation fails")

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

    def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        user_id: str,
        graph_id: str,
        graph_exec_id: str,
        **kwargs
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

            result: list[dict[str, Any]] = client.search(
                input_data.query, version="v2", filters=filters
            )
            yield "memories", result

        except Exception as e:
            yield "error", str(e)


class GetAllMemoriesBlock(Block, Mem0Base):
    """Block for retrieving all memories from Mem0"""

    class Input(BlockSchema):
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
        limit_memory_to_run: bool = SchemaField(
            description="Limit the memory to the run", default=False
        )
        limit_memory_to_agent: bool = SchemaField(
            description="Limit the memory to the agent", default=False
        )

    class Output(BlockSchema):
        memories: Any = SchemaField(description="List of memories")
        error: str = SchemaField(description="Error message if operation fails")

    def __init__(self):
        super().__init__(
            id="45aee5bf-4767-45d1-a28b-e01c5aae9fc1",
            description="Retrieve all memories from Mem0 with pagination",
            input_schema=GetAllMemoriesBlock.Input,
            output_schema=GetAllMemoriesBlock.Output,
            test_input={
                "user_id": "test_user",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                ("memories", [{"id": "test-memory", "content": "test content"}]),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={"_get_client": lambda credentials: MockMemoryClient()},
        )

    def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        user_id: str,
        graph_id: str,
        graph_exec_id: str,
        **kwargs
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

            memories: list[dict[str, Any]] = client.get_all(
                filters=filters,
                version="v2",
            )

            yield "memories", memories

        except Exception as e:
            yield "error", str(e)


# Mock client for testing
class MockMemoryClient:
    """Mock Mem0 client for testing"""

    def add(self, *args, **kwargs):
        return {"memory_id": "test-memory-id", "status": "success"}

    def search(self, *args, **kwargs) -> list[dict[str, str]]:
        return [{"id": "test-memory", "content": "test content"}]

    def get_all(self, *args, **kwargs) -> list[dict[str, str]]:
        return [{"id": "test-memory", "content": "test content"}]
