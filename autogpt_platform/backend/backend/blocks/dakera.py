"""Blocks for Dakera — decay-weighted persistent memory for agents.

`Dakera <https://dakera.ai>`_ is a self-hosted memory server that stores agent
memories with access-weighted importance decay and retrieves them by semantic
recall. Because Dakera is self-hosted, each block takes the server ``host`` URL
(default ``http://localhost:3000``) alongside an API key credential.

Quick start (self-host with the public docker-compose, which also provisions the
object store)::

    git clone https://github.com/dakera-ai/dakera-deploy
    cd dakera-deploy && docker compose up -d   # server on :3000 + MinIO

    pip install dakera

The blocks below wrap the ``dakera`` Python SDK's ``store_memory`` and ``recall``
methods. Memories are namespaced per AutoGPT agent graph by default so each
agent keeps its own memory; set ``agent_id`` to share memory across agents.

See https://dakera.ai/docs for the full API reference.
"""

from typing import Any, Literal, Optional

from dakera import DakeraClient, RecalledMemory, RecallResponse
from pydantic import SecretStr

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName

# Default self-hosted Dakera server endpoint (Dakera listens on port 3000).
DEFAULT_HOST = "http://localhost:3000"

MemoryType = Literal["episodic", "semantic", "procedural", "working"]

DakeraCredentials = APIKeyCredentials
DakeraCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.DAKERA], Literal["api_key"]
]


def DakeraCredentialsField() -> DakeraCredentialsInput:
    """Create a Dakera API key credentials input field."""
    return CredentialsField(description="Dakera API key (looks like ``dk-...``)")


TEST_CREDENTIALS = APIKeyCredentials(
    id="0f9d81b5-a145-4c23-b87f-01d6bf37b678",
    provider="dakera",
    api_key=SecretStr("dk-mock-dakera-api-key"),
    title="Mock Dakera API key",
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


class DakeraBase:
    """Base class with shared utilities for Dakera blocks."""

    @staticmethod
    def _get_client(credentials: APIKeyCredentials, host: str) -> DakeraClient:
        """Get an initialized Dakera client for a self-hosted server."""
        return DakeraClient(
            base_url=host or DEFAULT_HOST,
            api_key=credentials.api_key.get_secret_value(),
        )

    @staticmethod
    def _resolve_agent_id(explicit: str, graph_id: str) -> str:
        """Namespace memories per agent graph unless overridden.

        Each AutoGPT agent graph maps to its own Dakera ``agent_id`` so memory
        does not leak between agents. Passing an explicit ``agent_id`` opts into
        memory shared across agents (e.g. a team-wide knowledge base).
        """
        return explicit.strip() or graph_id


class StoreMemoryBlock(Block, DakeraBase):
    """Store a memory in Dakera, namespaced to the running agent."""

    class Input(BlockSchemaInput):
        credentials: DakeraCredentialsInput = DakeraCredentialsField()
        content: str = SchemaField(
            description="The memory content to store.",
            advanced=False,
        )
        importance: Optional[float] = SchemaField(
            description="Importance score 0.0–1.0. Higher values decay slower.",
            default=None,
            ge=0.0,
            le=1.0,
        )
        memory_type: MemoryType = SchemaField(
            description="Kind of memory to store.",
            default="episodic",
        )
        tags: list[str] = SchemaField(
            description="Optional tags to attach to the memory.",
            default_factory=list,
            advanced=True,
        )
        agent_id: str = SchemaField(
            description=(
                "Dakera memory namespace. Defaults to this agent's graph so "
                "each agent keeps its own memory; set to share memory across "
                "agents."
            ),
            default="",
            advanced=True,
        )
        host: str = SchemaField(
            description="Base URL of your Dakera server.",
            default=DEFAULT_HOST,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        memory_id: str = SchemaField(description="ID of the stored memory.")
        memory: dict[str, Any] = SchemaField(description="The stored memory record.")

    def __init__(self):
        super().__init__(
            id="6b3c8a2e-2f4d-4c9a-9d1e-2a7b5c3e4f10",
            description="Store a memory in a self-hosted Dakera server.",
            categories={BlockCategory.DATA},
            input_schema=StoreMemoryBlock.Input,
            output_schema=StoreMemoryBlock.Output,
            test_input={
                "content": "The user prefers dark mode.",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                ("memory_id", "3f2a1c7d-0b9e-4d6a-8c1f-2e5b7a9d0c31"),
                (
                    "memory",
                    {
                        "id": "3f2a1c7d-0b9e-4d6a-8c1f-2e5b7a9d0c31",
                        "content": "The user prefers dark mode.",
                        "memory_type": "episodic",
                        "importance": 0.7,
                    },
                ),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={
                "_get_client": lambda credentials, host: MockDakeraClient(),
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        graph_id: str,
        graph_exec_id: str,
        **kwargs,
    ) -> BlockOutput:
        try:
            client = self._get_client(credentials, input_data.host)
            agent_id = self._resolve_agent_id(input_data.agent_id, graph_id)

            memory = client.store_memory(
                agent_id=agent_id,
                content=input_data.content,
                memory_type=input_data.memory_type,
                importance=input_data.importance,
                tags=input_data.tags or None,
                session_id=graph_exec_id,
            )

            yield "memory_id", memory.get("id", "")
            yield "memory", memory
        except Exception as e:
            yield "error", str(e)


class RecallMemoryBlock(Block, DakeraBase):
    """Recall memories from Dakera by semantic query."""

    class Input(BlockSchemaInput):
        credentials: DakeraCredentialsInput = DakeraCredentialsField()
        query: str = SchemaField(
            description="Semantic query used to recall relevant memories.",
            advanced=False,
        )
        top_k: int = SchemaField(
            description="Maximum number of memories to return.",
            default=5,
            ge=1,
            le=100,
        )
        min_importance: Optional[float] = SchemaField(
            description="Only recall memories at or above this importance (0.0–1.0).",
            default=None,
            ge=0.0,
            le=1.0,
        )
        memory_type: Optional[MemoryType] = SchemaField(
            description="Optionally restrict recall to a single memory type.",
            default=None,
            advanced=True,
        )
        agent_id: str = SchemaField(
            description=(
                "Dakera memory namespace. Defaults to this agent's graph; set "
                "to recall from a namespace shared across agents."
            ),
            default="",
            advanced=True,
        )
        host: str = SchemaField(
            description="Base URL of your Dakera server.",
            default=DEFAULT_HOST,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        memories: list[dict[str, Any]] = SchemaField(
            description="Recalled memories ordered by relevance."
        )
        count: int = SchemaField(description="Number of memories recalled.")

    def __init__(self):
        super().__init__(
            id="c4e91d5a-7b28-4f3c-a6d0-9e1f2b8c4a56",
            description="Recall memories from a self-hosted Dakera server.",
            categories={BlockCategory.DATA},
            input_schema=RecallMemoryBlock.Input,
            output_schema=RecallMemoryBlock.Output,
            test_input={
                "query": "user interface preferences",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                (
                    "memories",
                    [
                        {
                            "id": "3f2a1c7d-0b9e-4d6a-8c1f-2e5b7a9d0c31",
                            "content": "The user prefers dark mode.",
                            "memory_type": "episodic",
                            "importance": 0.7,
                            "score": 0.91,
                            "created_at": None,
                        }
                    ],
                ),
                ("count", 1),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={
                "_get_client": lambda credentials, host: MockDakeraClient(),
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        graph_id: str,
        **kwargs,
    ) -> BlockOutput:
        try:
            client = self._get_client(credentials, input_data.host)
            agent_id = self._resolve_agent_id(input_data.agent_id, graph_id)

            response: RecallResponse = client.recall(
                agent_id=agent_id,
                query=input_data.query,
                top_k=input_data.top_k,
                min_importance=input_data.min_importance,
                memory_type=input_data.memory_type,
            )

            memories = [
                {
                    "id": m.id,
                    "content": m.content,
                    "memory_type": m.memory_type,
                    "importance": m.importance,
                    "score": m.score,
                    "created_at": m.created_at,
                }
                for m in response.memories
            ]

            yield "memories", memories
            yield "count", len(memories)
        except Exception as e:
            yield "error", str(e)


class MockDakeraClient:
    """Mock Dakera client for block tests."""

    def store_memory(self, *args, **kwargs) -> dict[str, Any]:
        return {
            "id": "3f2a1c7d-0b9e-4d6a-8c1f-2e5b7a9d0c31",
            "content": kwargs.get("content", "test memory"),
            "memory_type": kwargs.get("memory_type", "episodic"),
            "importance": 0.7,
        }

    def recall(self, *args, **kwargs) -> RecallResponse:
        return RecallResponse(
            memories=[
                RecalledMemory(
                    id="3f2a1c7d-0b9e-4d6a-8c1f-2e5b7a9d0c31",
                    content="The user prefers dark mode.",
                    memory_type="episodic",
                    importance=0.7,
                    score=0.91,
                )
            ]
        )
