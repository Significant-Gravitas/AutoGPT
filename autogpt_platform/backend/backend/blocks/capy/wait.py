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
from ._pull_requests import find_pull_request_url
from ._testdata import TEST_IDLE_THREAD, TEST_MESSAGES, TEST_THREAD
from ._types import ACTIVE_THREAD_STATUSES, STOPPED_THREAD_STATUSES, Message, Thread
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
        after_message_id: str = SchemaField(
            description=(
                "The message_id Capy Send Message returned. The wait then ends "
                "on the agent's reply to that message, never an earlier one."
            ),
            default="",
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
        thread_url: str = SchemaField(
            description="The thread in the Capy app, where its work shows live"
        )
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
                "question, or the pull request link. With after_message_id, "
                "only a reply to that message; empty until it answers."
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
        pull_request_url: str = SchemaField(
            description=(
                "The newest GitHub pull request link in the agent's recent "
                "replies, ready for the GitHub pull request blocks. Only "
                "emitted when the agent has linked one."
            )
        )

    def __init__(self):
        super().__init__(
            id="1a07ce26-5ae6-411b-9f6c-828ce56c8e03",
            description=(
                "Waits for a Capy thread to finish (the agent delivered, asked a "
                "question or failed) and returns its status, latest reply and a "
                "link to the thread. Returns early when the timeout runs out; "
                "call it again to keep waiting. After Capy Send Message, pass "
                "its message_id so the wait ends on the reply to that message."
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
                ("thread_url", TEST_IDLE_THREAD.url),
                ("status", "idle"),
                ("finished", True),
                ("needs_you", False),
                ("last_reply", TEST_MESSAGES[-1].text),
                ("model_id", "supergrok/grok-4.5"),
                ("billed_via", "SuperGrok subscription"),
                ("pull_request_url", "https://github.com/acme/app/pull/12"),
            ],
            test_mock={
                "wait_for_thread": lambda *args, **kwargs: (
                    TEST_IDLE_THREAD,
                    TEST_MESSAGES[-1].text,
                    True,
                    "https://github.com/acme/app/pull/12",
                )
            },
            effect=BlockEffect.READ,
        )

    @staticmethod
    async def wait_for_thread(
        credentials: APIKeyCredentials, input_data: "CapyWaitForThreadBlock.Input"
    ) -> tuple[Thread, str, bool, str]:
        """Return the thread's latest state, its last reply, whether it
        finished, and the newest pull request link it replied with."""
        client = CapyClient(credentials)
        thread, messages, finished = await _poll(
            client,
            input_data.thread_id,
            input_data.timeout_seconds,
            input_data.poll_interval_seconds,
            input_data.after_message_id,
        )
        thread, reply = _with_reply(thread, messages, input_data.after_message_id)
        pr_url = await find_pull_request_url(client, thread.project_id, messages)
        return thread, reply, finished, pr_url

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        thread, last_reply, finished, pr_url = await self.wait_for_thread(
            credentials, input_data
        )
        yield "thread", thread
        yield "thread_url", thread.url
        yield "status", thread.status
        yield "finished", finished
        yield "needs_you", thread.needs_you
        yield "last_reply", last_reply
        yield "model_id", thread.last_model_id or ""
        yield "billed_via", billed_via(thread.last_model_id)
        if pr_url:
            yield "pull_request_url", pr_url


async def _poll(
    client: CapyClient,
    thread_id: str,
    timeout_seconds: int,
    poll_interval: int,
    after_message_id: str = "",
) -> tuple[Thread, list[Message], bool]:
    deadline = time.monotonic() + timeout_seconds
    stopped_checks = 0
    while True:
        thread = await client.get_thread(thread_id)
        finished = thread.needs_you
        if not finished and thread.status not in ACTIVE_THREAD_STATUSES:
            page = await client.newest_messages(thread_id, limit=50)
            # A thread keeps its previous status (usually idle) for a moment
            # after it is created or sent a message, before the agent picks
            # the message up, so an idle status alone doesn't mean it answered.
            if _answered(page.items, after_message_id):
                return thread, page.items, True
            # A failed or archived thread won't pick the message up on its
            # own, so a second check that still finds it unanswered ends the
            # wait rather than the timeout.
            if thread.status in STOPPED_THREAD_STATUSES:
                stopped_checks += 1
                if stopped_checks >= 2:
                    return thread, page.items, True
        remaining = deadline - time.monotonic()
        if finished or remaining <= 0:
            break
        await asyncio.sleep(min(poll_interval, remaining))
    page = await client.newest_messages(thread_id, limit=50)
    return thread, page.items, finished


def _answered(messages: list[Message], after_message_id: str) -> bool:
    """Whether the agent has replied to the message the wait is about.

    Transcript IDs are ULIDs, so they sort by time: a reply to the given
    message sorts after it. Without one, the latest user entry is the message,
    and it is answered once anything follows it.
    """
    if after_message_id:
        return any(
            m.source == "assistant" and m.text and m.id > after_message_id
            for m in messages
        )
    return not messages or messages[-1].source != "user"


def _with_reply(
    thread: Thread, messages: list[Message], after_message_id: str = ""
) -> tuple[Thread, str]:
    """Pair the thread with its last reply, taking the model from that reply
    when it answers the message being waited on.

    Capy stamps each assistant message with the model that wrote it, while the
    thread's ``lastModelId`` can still name the previous model right after a
    switch; the reply is the reliable record. A reply from an earlier turn
    says nothing about the model working on this one.
    """
    if after_message_id:
        reply = _last_assistant([m for m in messages if m.id > after_message_id])
        current = reply is not None
    else:
        reply = _last_assistant(messages)
        current = _replied_since_last_user_entry(messages)
    if reply is None:
        return thread, ""
    if reply.model and current:
        thread = thread.model_copy(update={"last_model_id": reply.model})
    return thread, reply.text


def _replied_since_last_user_entry(messages: list[Message]) -> bool:
    for message in reversed(messages):
        if message.source == "user":
            return False
        if message.source == "assistant" and message.text:
            return True
    return False
