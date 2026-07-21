import time
from enum import Enum
from typing import Literal, Optional

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.desktop._api import WORKSPACE_PATH, DesktopSession
from backend.blocks.desktop._common import (
    CREDENTIALS_FIELD_DESCRIPTION,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
)
from backend.blocks.desktop._cost import CostMeter, build_cost_meter
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName


class FileOperation(str, Enum):
    READ = "read"
    WRITE = "write"
    LIST = "list"


class DesktopFileBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.E2B], Literal["api_key"]
        ] = CredentialsField(description=CREDENTIALS_FIELD_DESCRIPTION)
        sandbox_id: str = SchemaField(
            description="ID of the desktop sandbox to access files in"
        )
        operation: FileOperation = SchemaField(
            description="File operation to perform",
            default=FileOperation.READ,
            advanced=False,
        )
        path: str = SchemaField(
            description=(
                "File or directory path inside the sandbox; the persistent "
                f"workspace lives at {WORKSPACE_PATH}"
            ),
            placeholder=f"{WORKSPACE_PATH}/notes.txt",
        )
        content: str = SchemaField(
            description="Content to write (for the 'write' operation)", default=""
        )

    class Output(BlockSchemaOutput):
        content: str = SchemaField(description="File content (for 'read')")
        entries: list[str] = SchemaField(description="Directory entries (for 'list')")
        path: str = SchemaField(description="Path that was operated on")
        cost_meter: CostMeter = SchemaField(
            description="Estimated provider cost telemetry for this block run"
        )

    def __init__(self):
        super().__init__(
            id="a1e2b001-0004-4000-8000-de5c704b0004",
            description=(
                "Reads, writes, or lists files inside an interactive desktop "
                "sandbox, including its persistent workspace."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=DesktopFileBlock.Input,
            output_schema=DesktopFileBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "sandbox_id": "test-sandbox-id",
                "operation": FileOperation.READ.value,
                "path": "/home/user/workspace/notes.txt",
            },
            test_output=[
                ("content", "hello"),
                ("path", "/home/user/workspace/notes.txt"),
                ("cost_meter", lambda v: v["provider"] == "e2b"),
            ],
            test_mock={"file_op": lambda *args, **kwargs: ("hello", None)},
        )

    async def file_op(
        self,
        api_key: str,
        sandbox_id: str,
        operation: FileOperation,
        path: str,
        content: str,
    ) -> tuple[Optional[str], Optional[list[str]]]:
        session = await DesktopSession.connect(sandbox_id, api_key)
        if operation == FileOperation.READ:
            data = await session.sandbox.files.read(path)
            return data, None
        if operation == FileOperation.WRITE:
            await session.sandbox.files.write(path, content)
            return None, None
        entries = await session.sandbox.files.list(path)
        return None, [entry.name for entry in entries]

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        start = time.monotonic()
        try:
            content, entries = await self.file_op(
                api_key=credentials.api_key.get_secret_value(),
                sandbox_id=input_data.sandbox_id,
                operation=input_data.operation,
                path=input_data.path,
                content=input_data.content,
            )
            if content is not None:
                yield "content", content
            if entries is not None:
                yield "entries", entries
            yield "path", input_data.path
            yield "cost_meter", build_cost_meter(
                input_data.sandbox_id, time.monotonic() - start
            ).model_dump()
        except Exception as e:
            yield "error", str(e)
