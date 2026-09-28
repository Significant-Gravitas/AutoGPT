"""Blocks that start, inspect, wait on and archive Capy agent threads."""

import asyncio
import time

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
from ._testdata import TEST_IDLE_THREAD, TEST_MESSAGES, TEST_PROJECT, TEST_THREAD
from ._types import (
    ACTIVE_THREAD_STATUSES,
    DEFAULT_MODEL_ID,
    MachineSize,
    Message,
    ReasoningEffort,
    Thread,
)

# The block executor caps a run at 30 minutes; stay well inside it.
MAX_WAIT_SECONDS = 25 * 60


def _thread_id_field() -> str:
    return SchemaField(
        description="The Capy thread ID (starts with jam_)",
        placeholder="jam_01M2KY54H0CZC9S7M4DAQZYN7M",
    )


def _last_assistant_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.source == "assistant" and message.text:
            return message.text
    return ""


class CapyCreateThreadBlock(Block):
    """Start a Capy agent thread: a cloud coding agent working on a project's repos."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = capy_credentials_field()
        project_id: str = SchemaField(
            description="The Capy project to run in (see Capy List Projects)"
        )
        message: str = SchemaField(
            description=(
                "The task for the agent, written as you would brief an engineer: "
                "the goal, where to look, what done looks like, and whether to "
                "open a pull request"
            ),
        )
        title: str = SchemaField(
            description="Thread title. Leave empty to let Capy name it.",
            default="",
        )
        model_id: str = SchemaField(
            description=(
                "Capy model ID (see docs.capy.ai/models-and-pricing), e.g. "
                "meta/muse-spark-1.3 or openai/gpt-6-astra. Clear it to use "
                "the project's default model."
            ),
            default=DEFAULT_MODEL_ID,
            advanced=True,
        )
        reasoning: ReasoningEffort = SchemaField(
            description="Reasoning effort for the chosen model. Needs model_id.",
            default=ReasoningEffort.DEFAULT,
            advanced=True,
        )
        machine_size: MachineSize = SchemaField(
            description="Machine size for the agent's VM. Empty uses Capy's default.",
            default=MachineSize.DEFAULT,
            advanced=True,
        )
        request_id: str = SchemaField(
            description=(
                "Idempotency key. Re-sending the same request_id returns the "
                "thread it already created instead of starting a second run. "
                "Leave empty to generate one."
            ),
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        thread: Thread = SchemaField(description="The created thread")
        thread_id: str = SchemaField(description="ID of the created thread")
        status: str = SchemaField(description="The thread's status right after start")

    def __init__(self):
        super().__init__(
            id="f8ffe722-ceba-4fb3-81df-a3fd465ff29c",
            description=(
                "Starts a Capy cloud coding agent on a task in one of your Capy "
                "projects, such as fixing a bug, writing a feature or opening a "
                "pull request. Returns immediately with the thread ID; use Capy "
                "Wait For Thread to wait for the result. The run bills your "
                "Capy organization."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.AGENT},
            input_schema=CapyCreateThreadBlock.Input,
            output_schema=CapyCreateThreadBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "project_id": TEST_PROJECT.id,
                "message": "Upgrade the CI pipeline to Node 24 and open a PR.",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("thread", TEST_THREAD),
                ("thread_id", TEST_THREAD.id),
                ("status", "working"),
            ],
            test_mock={"create_thread": lambda *args, **kwargs: TEST_THREAD},
            effect=BlockEffect.EXTERNAL,
        )

    @staticmethod
    async def create_thread(
        credentials: APIKeyCredentials, input_data: "CapyCreateThreadBlock.Input"
    ) -> Thread:
        return await CapyClient(credentials).create_thread(
            project_id=input_data.project_id,
            message=input_data.message,
            title=input_data.title,
            model_id=input_data.model_id,
            reasoning=input_data.reasoning.value,
            machine_size=input_data.machine_size.value,
            request_id=input_data.request_id,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        thread = await self.create_thread(credentials, input_data)
        yield "thread", thread
        yield "thread_id", thread.id
        yield "status", thread.status


class CapyGetThreadBlock(Block):
    """Read a Capy thread's status, title and credit usage."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = capy_credentials_field()
        thread_id: str = _thread_id_field()

    class Output(BlockSchemaOutput):
        thread: Thread = SchemaField(description="The thread")
        status: str = SchemaField(
            description="working, waiting, idle, failed or archived"
        )
        is_active: bool = SchemaField(
            description="True while the agent is still working on the thread"
        )
        needs_you: bool = SchemaField(
            description="True when the agent is waiting on an answer from a person"
        )

    def __init__(self):
        super().__init__(
            id="2e4088c6-8684-409f-8f67-8e2dd87ba6c4",
            description=(
                "Gets a Capy thread's current status, title and credit usage, "
                "and whether it is still working or needs an answer."
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
                ("status", "working"),
                ("is_active", True),
                ("needs_you", False),
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
        yield "status", thread.status
        yield "is_active", thread.status in ACTIVE_THREAD_STATUSES
        yield "needs_you", thread.needs_you


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


class CapyWaitForThreadBlock(Block):
    """Poll a Capy thread until the agent stops working, then return its last reply."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = capy_credentials_field()
        thread_id: str = _thread_id_field()
        timeout_seconds: int = SchemaField(
            description=(
                "How long to wait before returning the current state. Call the "
                "block again to keep waiting. Keep it at 240 or less when "
                "running from chat, which cancels a block call after 5 minutes."
            ),
            default=240,
            ge=0,
            le=MAX_WAIT_SECONDS,
        )
        poll_interval_seconds: int = SchemaField(
            description="Seconds between status checks",
            default=15,
            ge=5,
            le=300,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        thread: Thread = SchemaField(description="The thread's latest state")
        status: str = SchemaField(
            description="working, waiting, idle, failed or archived"
        )
        finished: bool = SchemaField(
            description=(
                "True when the agent stopped working (it delivered, asked a "
                "question or failed); false when the timeout ran out first"
            )
        )
        needs_you: bool = SchemaField(
            description="True when the agent is waiting on an answer from a person"
        )
        last_reply: str = SchemaField(
            description=(
                "The agent's most recent reply, which carries its result, its "
                "question, or the pull request link"
            )
        )

    def __init__(self):
        super().__init__(
            id="1a07ce26-5ae6-411b-9f6c-828ce56c8e03",
            description=(
                "Waits for a Capy thread to finish (the agent delivered, asked a "
                "question or failed) and returns its status and latest reply. "
                "Returns early when the timeout runs out; call it again to keep "
                "waiting."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=CapyWaitForThreadBlock.Input,
            output_schema=CapyWaitForThreadBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "thread_id": TEST_THREAD.id,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("thread", TEST_IDLE_THREAD),
                ("status", "idle"),
                ("finished", True),
                ("needs_you", False),
                ("last_reply", TEST_MESSAGES[-1].text),
            ],
            test_mock={
                "wait_for_thread": lambda *args, **kwargs: (
                    TEST_IDLE_THREAD,
                    TEST_MESSAGES[-1].text,
                    True,
                )
            },
            effect=BlockEffect.READ,
        )

    @staticmethod
    async def wait_for_thread(
        credentials: APIKeyCredentials,
        thread_id: str,
        timeout_seconds: int,
        poll_interval_seconds: int,
    ) -> tuple[Thread, str, bool]:
        """Return the thread's latest state, its last reply, and whether it finished."""
        client = CapyClient(credentials)
        deadline = time.monotonic() + timeout_seconds
        while True:
            thread = await client.get_thread(thread_id)
            finished = thread.needs_you
            if not finished and thread.status not in ACTIVE_THREAD_STATUSES:
                page = await client.newest_messages(thread_id, limit=20)
                # A thread reads idle for a moment after it is created or sent
                # a message, before the agent picks the message up. The agent
                # has only finished once something follows the last user entry.
                finished = not page.items or page.items[-1].source != "user"
                if finished:
                    return thread, _last_assistant_text(page.items), True
            remaining = deadline - time.monotonic()
            if finished or remaining <= 0:
                break
            await asyncio.sleep(min(poll_interval_seconds, remaining))
        page = await client.newest_messages(thread_id, limit=20)
        return thread, _last_assistant_text(page.items), finished

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        thread, last_reply, finished = await self.wait_for_thread(
            credentials,
            input_data.thread_id,
            input_data.timeout_seconds,
            input_data.poll_interval_seconds,
        )
        yield "thread", thread
        yield "status", thread.status
        yield "finished", finished
        yield "needs_you", thread.needs_you
        yield "last_reply", last_reply


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
