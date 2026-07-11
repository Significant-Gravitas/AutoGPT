"""Blocks for Dakera — decay-weighted persistent memory for agents.

`Dakera <https://dakera.ai>`_ is a self-hosted memory server that stores agent
memories with access-weighted importance decay and retrieves them by semantic
recall. Because Dakera is self-hosted, the server URL varies per user, so these
blocks use **host-scoped credentials**: the server ``host`` (e.g.
``http://localhost:3000``) is stored together with the API key. Binding the key
to its host means a graph cannot redirect a stored key to an arbitrary server.

The host is additionally validated against the platform's SSRF egress guard
(``backend.util.request.validate_url_host``) before every call, so a credential
cannot be pointed at private, link-local, or cloud-metadata addresses. On a
shared deployment the Dakera server must therefore be reachable at a
non-private address.

Quick start (self-host with the public docker-compose, which also provisions the
object store)::

    git clone https://github.com/dakera-ai/dakera-deploy
    cd dakera-deploy && docker compose up -d   # server on :3000 + MinIO

    pip install dakera

Then add a Dakera credential in AutoGPT with ``host`` set to your server URL and
an ``Authorization`` header of ``Bearer dk-...`` (how the Dakera SDK
authenticates).

The blocks below wrap the ``dakera`` Python SDK's ``store_memory`` and ``recall``
methods. Memories are namespaced per user **and** agent graph by default
(``"{user_id}:{graph_id}"``) so memory never leaks across users or agents; set
``agent_id`` to opt into a namespace shared across users/agents.

See https://dakera.ai/docs for the full API reference.
"""

import asyncio
import logging
from typing import Any, Literal, Optional
from urllib.parse import urlparse

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
    CredentialsField,
    CredentialsMetaInput,
    HostScopedCredentials,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.request import validate_url_host

logger = logging.getLogger(__name__)

# Explicit client timeout (the SDK also defaults to 30s) so a slow or
# unreachable self-hosted server cannot hang a node's execution.
DEFAULT_TIMEOUT = 30.0

# Reference self-hosted Dakera endpoint (Dakera listens on port 3000). This is
# only used by the mock test credential below — real runs require the user to
# set their own ``host`` on the credential (``_get_client`` fails fast if unset),
# so this default is never silently used against the wrong service.
DEFAULT_HOST = "http://localhost:3000"

# Hosts for which a plaintext ``http://`` connection does not leave the machine.
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}

MemoryType = Literal["episodic", "semantic", "procedural", "working"]

# Dakera is self-hosted, so the server URL is user-specific. Host-scoped
# credentials bind the API key to a host so it cannot be sent elsewhere.
DakeraCredentials = HostScopedCredentials
DakeraCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.DAKERA], Literal["host_scoped"]
]


def DakeraCredentialsField() -> DakeraCredentialsInput:
    """Create a Dakera host-scoped credentials input field."""
    return CredentialsField(
        description=(
            "Dakera server host plus API key. Set ``host`` to your server URL "
            "(e.g. ``http://localhost:3000``) and add an ``Authorization`` "
            "header of ``Bearer dk-...``. On a shared deployment the host must "
            "be a non-private address (the platform's SSRF guard rejects "
            "private/link-local/metadata hosts)."
        )
    )


TEST_CREDENTIALS = HostScopedCredentials(
    id="0f9d81b5-a145-4c23-b87f-01d6bf37b678",
    provider="dakera",
    host=DEFAULT_HOST,
    headers={"Authorization": SecretStr("Bearer dk-mock-dakera-api-key")},
    title="Mock Dakera credentials",
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}

# Fields copied verbatim from the (user-controlled) Dakera server response into
# block outputs. Whitelisting avoids passing an arbitrary server payload — the
# potential exfiltration channel if a host were pointed at an internal service —
# straight into the graph, and keeps Store/Recall output shapes consistent.
_MEMORY_FIELDS = ("id", "content", "memory_type", "importance", "created_at")


class DakeraBase:
    """Base class with shared utilities for Dakera blocks."""

    @staticmethod
    async def _validate_host(host: str) -> None:
        """Reject hosts that resolve to private/link-local/metadata addresses.

        The credential ``host`` is user-controlled and the executor returns the
        server's response into the graph, so an unguarded client would be an
        SSRF vector (e.g. ``http://169.254.169.254``). Delegates to the same
        egress guard the native HTTP block uses; raises ``ValueError`` if the
        host is disallowed.
        """
        await validate_url_host(host)

    @staticmethod
    def _get_client(credentials: HostScopedCredentials) -> DakeraClient:
        """Get an initialized Dakera client for the credential's host.

        The server URL comes from the credential (not a graph input) and the
        API key travels as the credential's ``Authorization`` header, so the
        key is always bound to the host it was issued for. A missing host is a
        misconfiguration (a stored key with no bound server), so fail fast
        rather than silently sending the key to ``localhost``.
        """
        if not credentials.host:
            raise ValueError(
                "Dakera credential is missing a host — set the server URL "
                "(e.g. http://localhost:3000) on the credential."
            )
        parsed = urlparse(credentials.host)
        if parsed.scheme == "http" and parsed.hostname not in _LOOPBACK_HOSTS:
            logger.warning(
                "Dakera host %s uses http://; the Bearer API key will be sent "
                "in cleartext. Use https:// for non-loopback hosts.",
                parsed.hostname,
            )
        return DakeraClient(
            base_url=credentials.host,
            headers=credentials.get_headers_dict(),
            timeout=DEFAULT_TIMEOUT,
        )

    @staticmethod
    def _resolve_agent_id(explicit: str, graph_id: str, user_id: str) -> str:
        """Namespace memories per user and agent graph unless overridden.

        By default each ``(user, agent graph)`` pair maps to its own Dakera
        ``agent_id`` (``"{user_id}:{graph_id}"``) so memory never leaks between
        users or agents — even on a Dakera server shared across tenants. Passing
        an explicit ``agent_id`` opts into a namespace shared across users/agents
        (e.g. a team-wide knowledge base).
        """
        explicit = explicit.strip()
        if explicit:
            return explicit
        return f"{user_id}:{graph_id}"

    @staticmethod
    def _normalize_memory(record: dict[str, Any]) -> dict[str, Any]:
        """Whitelist known fields from a raw server memory record."""
        return {field: record.get(field) for field in _MEMORY_FIELDS}


class StoreDakeraMemoryBlock(Block, DakeraBase):
    """Store a memory in Dakera, namespaced to the running user and agent."""

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
                "Dakera memory namespace. Defaults to this user + agent graph "
                "so each keeps its own memory; set to share memory across "
                "users/agents."
            ),
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        memory_id: str = SchemaField(description="ID of the stored memory.")
        memory: dict[str, Any] = SchemaField(description="The stored memory record.")
        error: str = SchemaField(
            description="Error message if the store operation failed."
        )

    def __init__(self):
        super().__init__(
            id="6b3c8a2e-2f4d-4c9a-9d1e-2a7b5c3e4f10",
            description="Store a memory in a self-hosted Dakera server.",
            categories={BlockCategory.DATA},
            input_schema=StoreDakeraMemoryBlock.Input,
            output_schema=StoreDakeraMemoryBlock.Output,
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
                        "created_at": None,
                    },
                ),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={
                "_validate_host": lambda host: None,
                "_get_client": lambda credentials: MockDakeraClient(),
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: HostScopedCredentials,
        graph_id: str,
        graph_exec_id: str,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        try:
            await self._validate_host(credentials.host)
            client = self._get_client(credentials)
            agent_id = self._resolve_agent_id(input_data.agent_id, graph_id, user_id)

            # store_memory is a synchronous requests-based SDK call; offload it
            # so it does not block the shared node execution event loop.
            record = await asyncio.to_thread(
                client.store_memory,
                agent_id=agent_id,
                content=input_data.content,
                memory_type=input_data.memory_type,
                importance=input_data.importance,
                tags=input_data.tags or None,
                session_id=graph_exec_id,
            )

            memory_id = record.get("id")
            if not memory_id:
                raise ValueError("Dakera store response did not include a memory id.")

            yield "memory_id", memory_id
            yield "memory", self._normalize_memory(record)
        except Exception as e:
            yield "error", str(e)


class RecallDakeraMemoryBlock(Block, DakeraBase):
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
                "Dakera memory namespace. Defaults to this user + agent graph; "
                "set to recall from a namespace shared across users/agents."
            ),
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        memories: list[dict[str, Any]] = SchemaField(
            description="Recalled memories ordered by relevance."
        )
        count: int = SchemaField(description="Number of memories recalled.")
        error: str = SchemaField(
            description="Error message if the recall operation failed."
        )

    def __init__(self):
        super().__init__(
            id="c4e91d5a-7b28-4f3c-a6d0-9e1f2b8c4a56",
            description="Recall memories from a self-hosted Dakera server.",
            categories={BlockCategory.DATA},
            input_schema=RecallDakeraMemoryBlock.Input,
            output_schema=RecallDakeraMemoryBlock.Output,
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
                            "created_at": None,
                            "score": 0.91,
                        }
                    ],
                ),
                ("count", 1),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={
                "_validate_host": lambda host: None,
                "_get_client": lambda credentials: MockDakeraClient(),
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: HostScopedCredentials,
        graph_id: str,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        try:
            await self._validate_host(credentials.host)
            client = self._get_client(credentials)
            agent_id = self._resolve_agent_id(input_data.agent_id, graph_id, user_id)

            # recall is a synchronous requests-based SDK call; offload it so it
            # does not block the shared node execution event loop. (The SDK's
            # recall has no session filter, so run-scoped recall is not exposed.)
            response: RecallResponse = await asyncio.to_thread(
                client.recall,
                agent_id=agent_id,
                query=input_data.query,
                top_k=input_data.top_k,
                min_importance=input_data.min_importance,
                memory_type=input_data.memory_type,
            )

            memories = [
                {**self._normalize_memory(vars(m)), "score": m.score}
                for m in response.memories
            ]

            yield "memories", memories
            yield "count", len(memories)
        except Exception as e:
            yield "error", str(e)


class MockDakeraClient:
    """Mock Dakera client for block tests.

    ``store_memory`` mirrors the real SDK, which returns a plain ``dict`` — unlike
    ``recall``, which returns typed ``RecallResponse``/``RecalledMemory`` objects.
    """

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
