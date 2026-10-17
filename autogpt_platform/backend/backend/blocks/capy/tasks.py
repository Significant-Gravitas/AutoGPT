"""Block that lists the subagent tasks a Capy thread fanned its work out to."""

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
from ._testdata import TEST_TASK, TEST_THREAD
from ._types import Task
from .threads import _thread_id_field


class CapyListThreadTasksBlock(Block):
    """List a Capy thread's task tree, parents before children."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = capy_credentials_field()
        thread_id: str = _thread_id_field()
        limit: int = SchemaField(
            description="Maximum number of tasks to return",
            default=50,
            ge=1,
            le=200,
        )
        cursor: str = SchemaField(
            description="Paging cursor from a previous call's next_cursor",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        tasks: list[Task] = SchemaField(
            description=(
                "Tasks depth-first; task_path is the dotted address from the "
                "thread root and usage is each task's own subtree spend"
            )
        )
        next_cursor: str = SchemaField(
            description="Pass back as cursor for the next page; empty on the last"
        )

    def __init__(self):
        super().__init__(
            id="b7576f85-d0f2-4035-9116-19dfb1e520c5",
            description=(
                "Lists the subagent tasks a Capy thread fanned its work out to, "
                "with each task's status and credit spend. Read-only: steer a "
                "task by messaging its thread."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=CapyListThreadTasksBlock.Input,
            output_schema=CapyListThreadTasksBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "thread_id": TEST_THREAD.id,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("tasks", [TEST_TASK]), ("next_cursor", "")],
            test_mock={"list_tasks": lambda *args, **kwargs: ([TEST_TASK], None)},
            effect=BlockEffect.READ,
        )

    @staticmethod
    async def list_tasks(
        credentials: APIKeyCredentials, thread_id: str, limit: int, cursor: str
    ) -> tuple[list[Task], str | None]:
        return await CapyClient(credentials).list_tasks(thread_id, limit, cursor)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        tasks, cursor = await self.list_tasks(
            credentials, input_data.thread_id, input_data.limit, input_data.cursor
        )
        yield "tasks", tasks
        yield "next_cursor", cursor or ""
