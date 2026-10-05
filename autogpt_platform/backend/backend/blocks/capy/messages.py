"""Blocks that read a Capy thread's transcript and talk to its agent."""

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

from ._api import CapyClient, with_capy_balance_fallback
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, capy_credentials_field
from ._models import (
    ModelRoute,
    capy_balance_fallback_field,
    model_route_field,
    resolve_model_id,
)
from ._testdata import TEST_MESSAGES, TEST_RECEIPT, TEST_THREAD
from ._types import MessageDelivery, MessagePage, MessageReceipt, ReasoningEffort
from .threads import _last_assistant_text, _thread_id_field


class CapyListThreadMessagesBlock(Block):
    """Read a Capy thread's transcript: the brief, the agent's replies, and tool steps."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = capy_credentials_field()
        thread_id: str = _thread_id_field()
        limit: int = SchemaField(
            description="Maximum number of transcript entries to return",
            default=20,
            ge=1,
            le=200,
        )
        after_cursor: str = SchemaField(
            description=(
                "Return only entries after this cursor (a previous call's "
                "next_cursor), oldest first. Leave empty for the newest entries."
            ),
            default="",
            advanced=True,
        )
        before_cursor: str = SchemaField(
            description=(
                "Return the entries just before this cursor (a previous call's "
                "older_cursor), to read back through a long transcript"
            ),
            default="",
            advanced=True,
        )
        include_tool_steps: bool = SchemaField(
            description=(
                "Include the one-line tool activity entries alongside user and "
                "assistant messages"
            ),
            default=False,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        messages: list[dict] = SchemaField(
            description=(
                "Transcript entries, oldest first, each with id, source "
                "(user, assistant or tool), text, and created_at"
            )
        )
        last_reply: str = SchemaField(
            description="The agent's most recent reply on this page, if any"
        )
        next_cursor: str = SchemaField(
            description=(
                "Pass back as after_cursor to read only newer entries next time"
            )
        )
        older_cursor: str = SchemaField(
            description=(
                "Pass back as before_cursor to read the entries before this "
                "page; empty once the page starts at the transcript's beginning"
            )
        )

    def __init__(self):
        super().__init__(
            id="162bd732-53e2-4cf9-b357-10bf7f4cbea3",
            description=(
                "Reads a Capy thread's transcript: your brief, the agent's replies "
                "(including pull request links and questions), and optionally its "
                "tool steps. Returns the newest entries by default, and pages "
                "forward or back from a cursor."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=CapyListThreadMessagesBlock.Input,
            output_schema=CapyListThreadMessagesBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "thread_id": TEST_THREAD.id,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "messages",
                    [
                        m.model_dump(exclude_none=True)
                        for m in TEST_MESSAGES
                        if m.source != "tool"
                    ],
                ),
                ("last_reply", TEST_MESSAGES[-1].text),
                ("next_cursor", TEST_MESSAGES[-1].id),
                ("older_cursor", TEST_MESSAGES[0].id),
            ],
            test_mock={
                "list_messages": lambda *args, **kwargs: MessagePage(
                    items=TEST_MESSAGES,
                    cursor=TEST_MESSAGES[-1].id,
                    before_cursor=TEST_MESSAGES[0].id,
                )
            },
            effect=BlockEffect.READ,
        )

    @staticmethod
    async def list_messages(
        credentials: APIKeyCredentials, input_data: "CapyListThreadMessagesBlock.Input"
    ) -> MessagePage:
        client = CapyClient(credentials)
        if not (input_data.after_cursor or input_data.before_cursor):
            return await client.newest_messages(
                input_data.thread_id, limit=input_data.limit
            )
        return await client.list_messages(
            input_data.thread_id,
            limit=input_data.limit,
            after=input_data.after_cursor,
            before=input_data.before_cursor,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        page = await self.list_messages(credentials, input_data)
        items = [
            m for m in page.items if input_data.include_tool_steps or m.source != "tool"
        ]
        yield "messages", [m.model_dump(exclude_none=True) for m in items]
        yield "last_reply", _last_assistant_text(items)
        # The newest page reports no forward cursor, and an empty page past the
        # caller's cursor has none either. Fall back to the last message's
        # event ID (tool steps carry call_... IDs Capy rejects as cursors),
        # then to the caller's cursor, so polling never rewinds to the start
        # of the transcript.
        yield "next_cursor", (
            page.cursor
            or next((m.id for m in reversed(page.items) if m.source != "tool"), "")
            or input_data.after_cursor
        )
        yield "older_cursor", page.before_cursor or ""


class CapySendMessageBlock(Block):
    """Send a follow-up instruction or an answer to a Capy thread's agent."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = capy_credentials_field()
        thread_id: str = _thread_id_field()
        text: str = SchemaField(description="The message for the agent")
        delivery: MessageDelivery = SchemaField(
            description=(
                "interrupt stops the current work and handles this message now; "
                "steer folds it into the work in progress; queue waits until the "
                "current work finishes"
            ),
            default=MessageDelivery.INTERRUPT,
        )
        model_id: str = SchemaField(
            description=(
                "Switch the thread to this Capy model ID (or a bare name to "
                "combine with model_route). Empty keeps the thread's model."
            ),
            default="",
            advanced=True,
        )
        model_route: ModelRoute = model_route_field()
        fall_back_to_capy_balance: bool = capy_balance_fallback_field()
        reasoning: ReasoningEffort = SchemaField(
            description="Reasoning effort for model_id. Needs model_id.",
            default=ReasoningEffort.DEFAULT,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        message_id: str = SchemaField(
            description=(
                "ID of the admitted message. Pass it to Capy Wait For Thread as "
                "after_message_id, so the wait ends on the reply to this message."
            )
        )
        deduped: bool = SchemaField(
            description="True when Capy recognised this as a repeat of a message it already had"
        )
        model_id: str = SchemaField(
            description=(
                "The model the thread was switched to; empty when the thread "
                "kept its model"
            )
        )

    def __init__(self):
        super().__init__(
            id="4bcea811-e6ca-40ae-abdb-1e2b08f87d3c",
            description=(
                "Sends a message to the agent in a Capy thread: a follow-up "
                "instruction, a correction, or the answer to its question. The "
                "agent resumes work on it; wait for its reply with Capy Wait "
                "For Thread, passing message_id as after_message_id."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.AGENT},
            input_schema=CapySendMessageBlock.Input,
            output_schema=CapySendMessageBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "thread_id": TEST_THREAD.id,
                "text": "CI is green but the lockfile changed; explain why.",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("message_id", TEST_RECEIPT.id),
                ("deduped", False),
                ("model_id", ""),
            ],
            test_mock={"send_message": lambda *args, **kwargs: (TEST_RECEIPT, "")},
            effect=BlockEffect.EXTERNAL,
        )

    @staticmethod
    async def send_message(
        credentials: APIKeyCredentials, input_data: "CapySendMessageBlock.Input"
    ) -> tuple[MessageReceipt, str]:
        client = CapyClient(credentials)

        async def send(model_id: str) -> MessageReceipt:
            return await client.send_message(
                input_data.thread_id,
                text=input_data.text,
                delivery=input_data.delivery.value,
                model_id=model_id,
                reasoning=input_data.reasoning.value,
            )

        return await with_capy_balance_fallback(
            send,
            resolve_model_id(input_data.model_id, input_data.model_route),
            input_data.fall_back_to_capy_balance,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        receipt, model_id = await self.send_message(credentials, input_data)
        yield "message_id", receipt.id
        yield "deduped", receipt.deduped
        yield "model_id", model_id


class CapyInterruptThreadBlock(Block):
    """Stop a Capy thread's agent mid-run without sending it new instructions."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = capy_credentials_field()
        thread_id: str = _thread_id_field()

    class Output(BlockSchemaOutput):
        interrupted: bool = SchemaField(description="True once Capy accepted the stop")

    def __init__(self):
        super().__init__(
            id="284af9bf-78c0-4db1-92b5-f78a41147ea7",
            description=(
                "Stops the agent in a Capy thread mid-run, for example when it is "
                "heading the wrong way. The thread stays open for a new message."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=CapyInterruptThreadBlock.Input,
            output_schema=CapyInterruptThreadBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "thread_id": TEST_THREAD.id,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("interrupted", True)],
            test_mock={"interrupt_thread": lambda *args, **kwargs: TEST_RECEIPT},
            effect=BlockEffect.EXTERNAL,
        )

    @staticmethod
    async def interrupt_thread(credentials: APIKeyCredentials, thread_id: str):
        return await CapyClient(credentials).interrupt_thread(thread_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        await self.interrupt_thread(credentials, input_data.thread_id)
        yield "interrupted", True
