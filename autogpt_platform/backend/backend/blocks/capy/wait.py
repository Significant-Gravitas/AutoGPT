"""Block that waits for a Capy thread's agent to finish and returns its reply."""

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
from ._models import billed_via
from ._testdata import TEST_IDLE_THREAD, TEST_MESSAGES, TEST_THREAD
from ._types import ACTIVE_THREAD_STATUSES, Message, Thread
from .threads import MAX_WAIT_SECONDS, _last_assistant, _thread_id_field


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
                ("model_id", "supergrok/grok-4.5"),
                ("billed_via", "SuperGrok subscription"),
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
                    return _with_reply(thread, page.items, True)
            remaining = deadline - time.monotonic()
            if finished or remaining <= 0:
                break
            await asyncio.sleep(min(poll_interval_seconds, remaining))
        page = await client.newest_messages(thread_id, limit=20)
        return _with_reply(thread, page.items, finished)

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
        yield "model_id", thread.last_model_id or ""
        yield "billed_via", billed_via(thread.last_model_id)


def _with_reply(
    thread: Thread, messages: list[Message], finished: bool
) -> tuple[Thread, str, bool]:
    """Pair the thread with its last reply, taking the model from that reply.

    Capy stamps each assistant message with the model that wrote it, while the
    thread's ``lastModelId`` can still name the previous model right after a
    switch; the reply is the reliable record.
    """
    reply = _last_assistant(messages)
    if reply is None:
        return thread, "", finished
    if reply.model:
        thread = thread.model_copy(update={"last_model_id": reply.model})
    return thread, reply.text, finished
