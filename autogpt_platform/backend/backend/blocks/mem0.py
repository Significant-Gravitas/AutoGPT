from typing import Optional, Any, Union, Literal
from urllib.parse import urlencode
from pydantic import SecretStr
from backend.data.block import Block, BlockSchema, BlockOutput
from backend.data.model import (
    CredentialsField,
    APIKeyCredentials,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.request import requests


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


# Shared utilities for Mem0 blocks
class Mem0Base:
    """Base class with shared utilities for Mem0 blocks"""

    @staticmethod
    def _make_request(
        method: str, endpoint: str, data: dict[str, Any], credentials: APIKeyCredentials
    ) -> dict[str, Any]:
        """Make request to Mem0 API"""
        base_url = "https://api.mem0.ai/v1"
        headers = {
            "Authorization": f"Bearer {credentials.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        url = f"{base_url}/{endpoint}"

        response = requests.request(method=method, url=url, headers=headers, json=data)

        response.raise_for_status()
        return response.json()


class AddMemoryBlock(Block, Mem0Base):
    """Block for adding memories to Mem0"""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.MEM0], Literal["api_key"]
        ] = CredentialsField(description="Mem0 API key credentials")
        content: Union[str, list[dict[str, str]]] = SchemaField(
            description="Content to add - either a string or list of message objects"
        )
        metadata: Optional[dict[str, Any]] = SchemaField(
            description="Optional metadata for the memory", default=None
        )

    class Output(BlockSchema):
        memory_id: str = SchemaField(description="ID of the created memory")
        status: str = SchemaField(description="Status of the operation")
        error: str = SchemaField(description="Error message if operation fails")

    def __init__(self):
        super().__init__(
            id="dce97578-86be-45a4-ae50-f6de33fc935a",
            description="Add new memories to Mem0 with user segmentation",
            input_schema=AddMemoryBlock.Input,
            output_schema=AddMemoryBlock.Output,
            test_input={
                "user_id": "test_user",
                "content": [{"role": "user", "content": "I'm a vegetarian"}],
                "metadata": {"food": "vegetarian"},
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[("memory_id", "test-memory-id"), ("status", "success")],
            test_credentials=TEST_CREDENTIALS,
            test_mock={
                "_make_request": lambda method, endpoint, data, credentials: {
                    "status": "success",
                    "memory_id": "test-memory-id",
                }
            },
        )

    def run(
        self,
        input_data: Input,
        *,
        user_id: str,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            data = {
                "messages": (
                    input_data.content
                    if isinstance(input_data.content, list)
                    else [{"role": "user", "content": input_data.content}]
                ),
                "user_id": user_id,
                "output_format": "v1.1",
            }

            if input_data.metadata:
                data["metadata"] = input_data.metadata

            result = self._make_request("POST", "memory", data, credentials)

            yield "memory_id", result.get("memory_id")
            yield "status", result.get("status", "success")

        except Exception as e:
            yield "error", str(e)
            yield "status", "error"


class SearchMemoryBlock(Block, Mem0Base):
    """Block for searching memories in Mem0"""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.MEM0], Literal["api_key"]
        ] = CredentialsField(description="Mem0 API key credentials")
        query: str = SchemaField(description="Search query")
        metadata: Optional[dict[str, Any]] = SchemaField(
            description="Optional metadata filters", default=None
        )

    class Output(BlockSchema):
        memories: list[dict[str, Any]] = SchemaField(
            description="List of matching memories"
        )
        error: str = SchemaField(description="Error message if operation fails")

    def __init__(self):
        super().__init__(
            id="bd7c84e3-e073-4b75-810c-600886ec8a5b",
            description="Search memories in Mem0 by user",
            input_schema=SearchMemoryBlock.Input,
            output_schema=SearchMemoryBlock.Output,
            test_input={
                "user_id": "test_user",
                "query": "vegetarian preferences",
                "metadata": {"food": "vegetarian"},
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                ("memories", [{"id": "test-memory", "content": "test content"}])
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={
                "_make_request": lambda method, endpoint, data, credentials: {
                    "memories": [{"id": "test-memory", "content": "test content"}]
                }
            },
        )

    def run(
        self,
        input_data: Input,
        *,
        user_id: str,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            data: dict[str, Any] = {
                "query": input_data.query,
                "user_id": user_id,
                "output_format": "v1.1",
            }

            if input_data.metadata:
                data["metadata"] = input_data.metadata

            result = self._make_request("POST", "memory/search", data, credentials)

            yield "memories", result.get("memories", [])

        except Exception as e:
            yield "error", str(e)


class GetAllMemoriesBlock(Block, Mem0Base):
    """Block for retrieving all memories from Mem0"""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.MEM0], Literal["api_key"]
        ] = CredentialsField(description="Mem0 API key credentials")
        page: Optional[int] = SchemaField(description="Page number", default=1)
        page_size: Optional[int] = SchemaField(
            description="Number of items per page", default=50
        )
        categories: Optional[list[str]] = SchemaField(
            description="Filter by categories", default=None
        )

    class Output(BlockSchema):
        memories: list[dict[str, Any]] = SchemaField(description="List of memories")
        total_pages: int = SchemaField(description="Total number of pages")
        current_page: int = SchemaField(description="Current page number")
        error: str = SchemaField(description="Error message if operation fails")

    def __init__(self):
        super().__init__(
            id="45aee5bf-4767-45d1-a28b-e01c5aae9fc1",
            description="Retrieve all memories from Mem0 with pagination",
            input_schema=GetAllMemoriesBlock.Input,
            output_schema=GetAllMemoriesBlock.Output,
            test_input={
                "user_id": "test_user",
                "page": 1,
                "page_size": 50,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                ("memories", [{"id": "test-memory", "content": "test content"}]),
                ("total_pages", 1),
                ("current_page", 1),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={
                "_make_request": lambda method, endpoint, data, credentials: {
                    "memories": [{"id": "test-memory", "content": "test content"}],
                    "total_pages": 1,
                    "current_page": 1,
                }
            },
        )

    def run(
        self,
        input_data: Input,
        *,
        user_id: str,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            params = {
                "user_id": user_id,
                "page": input_data.page,
                "page_size": input_data.page_size,
                "output_format": "v1.1",
            }

            if input_data.categories:
                params["categories"] = input_data.categories

            result = self._make_request(
                "GET", f"memory?{urlencode(params)}", {}, credentials
            )

            yield "memories", result.get("memories", [])
            yield "total_pages", result.get("total_pages", 1)
            yield "current_page", result.get("current_page", 1)

        except Exception as e:
            yield "error", str(e)
