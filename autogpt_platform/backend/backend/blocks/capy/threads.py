"""Blocks that start, inspect, wait on and archive Capy agent threads."""

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import CapyClient
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, capy_credentials_field
from ._models import billed_via
from ._testdata import TEST_PROJECT, TEST_THREAD
from ._types import ACTIVE_THREAD_STATUSES, Message, Thread

# The block executor caps a run at 30 minutes; stay well inside it.
MAX_WAIT_SECONDS = 25 * 60


def _thread_id_field() -> str:
    return SchemaField(
        description="The Capy thread ID (starts with jam_)",
        placeholder="jam_01M2KY54H0CZC9S7M4DAQZYN7M",
    )


def _last_assistant_text(messages: list[Message]) -> str:
    reply = _last_assistant(messages)
    return reply.text if reply else ""


def _last_assistant(messages: list[Message]) -> Message | None:
    return next(
        (m for m in reversed(messages) if m.source == "assistant" and m.text), None
    )


class CapyGetThreadBlock(Block):
    """Read a Capy thread's status, title and credit usage."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = capy_credentials_field()
        thread_id: str = _thread_id_field()

    class Output(BlockSchemaOutput):
        thread: Thread = SchemaField(description="The thread")
        thread_url: str = SchemaField(
            description="The thread in the Capy app, where its work shows live"
        )
        status: str = SchemaField(
            description="working, waiting, idle, failed or archived"
        )
        is_active: bool = SchemaField(
            description="True while the agent is still working on the thread"
        )
        needs_you: bool = SchemaField(
            description="True when the agent is waiting on an answer from a person"
        )
        model_id: str = SchemaField(
            description="The model the agent last ran on, e.g. supergrok/grok-4.5"
        )
        billed_via: str = SchemaField(
            description=(
                "Who pays for that model: the Capy balance, or the linked "
                "provider (Codex, Copilot, SuperGrok, Azure)"
            )
        )

    def __init__(self):
        super().__init__(
            id="2e4088c6-8684-409f-8f67-8e2dd87ba6c4",
            description=(
                "Gets a Capy thread's current status, title and credit usage, "
                "whether it is still working or needs an answer, and a link to "
                "watch it."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=CapyGetThreadBlock.Input,
            output_schema=CapyGetThreadBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "thread_id": TEST_THREAD.id,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("thread", TEST_THREAD),
                ("thread_url", TEST_THREAD.url),
                ("status", "working"),
                ("is_active", True),
                ("needs_you", False),
                ("model_id", "supergrok/grok-4.5"),
                ("billed_via", "SuperGrok subscription"),
            ],
            test_mock={"get_thread": lambda *args, **kwargs: TEST_THREAD},
            effect=BlockEffect.READ,
        )

    @staticmethod
    async def get_thread(credentials: APIKeyCredentials, thread_id: str) -> Thread:
        return await CapyClient(credentials).get_thread(thread_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        thread = await self.get_thread(credentials, input_data.thread_id)
        yield "thread", thread
        yield "thread_url", thread.url
        yield "status", thread.status
        yield "is_active", thread.status in ACTIVE_THREAD_STATUSES
        yield "needs_you", thread.needs_you
        yield "model_id", thread.last_model_id or ""
        yield "billed_via", billed_via(thread.last_model_id)


class CapyListThreadsBlock(Block):
    """List the threads in a Capy project, most recently active first."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = capy_credentials_field()
        project_id: str = SchemaField(description="The Capy project to list")
        limit: int = SchemaField(
            description="Maximum number of threads to return",
            default=20,
            ge=1,
            le=100,
        )
        cursor: str = SchemaField(
            description="Paging cursor from a previous call's next_cursor",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        threads: list[Thread] = SchemaField(description="The threads on this page")
        thread: Thread = SchemaField(description="Each thread, one at a time")
        next_cursor: str = SchemaField(
            description="Pass back as cursor for the next page; empty on the last"
        )

    def __init__(self):
        super().__init__(
            id="98f03508-e4e0-474e-aace-306d3c887edc",
            description=(
                "Lists the agent threads in a Capy project with their status, "
                "most recently active first."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=CapyListThreadsBlock.Input,
            output_schema=CapyListThreadsBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "project_id": TEST_PROJECT.id,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("threads", [TEST_THREAD]),
                ("thread", TEST_THREAD),
                ("next_cursor", ""),
            ],
            test_mock={"list_threads": lambda *args, **kwargs: ([TEST_THREAD], None)},
            effect=BlockEffect.READ,
        )

    @staticmethod
    async def list_threads(
        credentials: APIKeyCredentials, project_id: str, limit: int, cursor: str
    ) -> tuple[list[Thread], str | None]:
        return await CapyClient(credentials).list_threads(project_id, limit, cursor)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        threads, cursor = await self.list_threads(
            credentials, input_data.project_id, input_data.limit, input_data.cursor
        )
        yield "threads", threads
        for thread in threads:
            yield "thread", thread
        yield "next_cursor", cursor or ""


class CapyArchiveThreadBlock(Block):
    """Archive a Capy thread once its work is done."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = capy_credentials_field()
        thread_id: str = _thread_id_field()

    class Output(BlockSchemaOutput):
        thread: Thread = SchemaField(description="The archived thread")
        archived: bool = SchemaField(description="Whether the thread is now archived")

    def __init__(self):
        super().__init__(
            id="0f1a1884-cb25-4403-87ec-61107ce51b02",
            description=(
                "Archives a Capy thread, taking it off the project board. "
                "Archived threads can be restored in the Capy app."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=CapyArchiveThreadBlock.Input,
            output_schema=CapyArchiveThreadBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "thread_id": TEST_THREAD.id,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("thread", TEST_THREAD.model_copy(update={"archived": True})),
                ("archived", True),
            ],
            test_mock={
                "archive_thread": lambda *args, **kwargs: TEST_THREAD.model_copy(
                    update={"archived": True}
                )
            },
            effect=BlockEffect.EXTERNAL,
        )

    @staticmethod
    async def archive_thread(credentials: APIKeyCredentials, thread_id: str) -> Thread:
        return await CapyClient(credentials).archive_thread(thread_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        thread = await self.archive_thread(credentials, input_data.thread_id)
        yield "thread", thread
        yield "archived", thread.archived
