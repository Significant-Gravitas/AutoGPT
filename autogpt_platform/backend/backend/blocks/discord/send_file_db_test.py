"""What Send Discord File attaches, and under which name (#17272).

The block's own ``run()`` with the context AutoPilot gives it, a file written
through ``WorkspaceManager`` to a real database and the real
``store_media_file``. Only the Discord client is replaced, by one that records
what the channel was sent."""

import base64
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import discord
import pytest
import pytest_asyncio
from prisma.models import UserWorkspace

from backend.blocks.discord.bot_blocks import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    SendDiscordFileBlock,
)
from backend.copilot.constants import COPILOT_SESSION_PREFIX
from backend.data import db
from backend.data.execution import ExecutionContext
from backend.util import file as file_module
from backend.util.workspace import WorkspaceManager
from backend.util.workspace_storage import LocalWorkspaceStorage

FILE_NAME = "Umsatz-November 🥨.csv"
CSV = "Monat;Umsatz\nNovember;12.345,67 €\n".encode()
PDF = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj <<>> endobj\ntrailer <<>>\n%%EOF\n"

# Each attachment the channel got, as (name, bytes).
Sent = list[tuple[str, bytes]]
# Runs the block on a file (and an optional filename): its status, and what was sent.
SendFile = Callable[..., Awaitable[tuple[str, Sent]]]


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "reference",
    [
        # What store_media_file(..., "for_block_output") hands AutoPilot.
        "workspace://{id}#text/csv",
        "workspace://{id}",
        "workspace:///{name}#text/csv",
        "workspace:///{name}",
    ],
    ids=["id-and-type", "id", "path-and-type", "path"],
)
async def test_a_workspace_file_goes_out_under_its_own_name(
    send: SendFile, chat: ExecutionContext, reference: str
):
    file_id = await _write_csv(chat)

    status, sent = await send(reference.format(id=file_id, name=FILE_NAME))

    assert status == "File sent successfully"
    assert sent == [(FILE_NAME, CSV)]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_filename_given_to_the_block_wins(
    send: SendFile, chat: ExecutionContext
):
    file_id = await _write_csv(chat)

    status, sent = await send(f"workspace://{file_id}#text/csv", "Bericht.csv")

    assert status == "File sent successfully"
    assert sent == [("Bericht.csv", CSV)]


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "data_uri, name, content",
    [
        (
            f"data:application/pdf;base64,{base64.b64encode(PDF).decode()}",
            "file.pdf",
            PDF,
        ),
        ("data:text/plain,Hallo%20Welt", "file.txt", b"Hallo Welt"),
        (
            "data:text/plain;charset=utf-8,Gr%C3%BC%C3%9Fe%20aus%20M%C3%BCnchen",
            "file.txt",
            "Grüße aus München".encode(),
        ),
        # RFC 2397: a data URI that names no media type holds text/plain.
        ("data:,Hallo", "file.txt", b"Hallo"),
    ],
    ids=["base64", "percent-encoded", "percent-encoded-utf8", "no-media-type"],
)
async def test_a_data_uri_goes_out_as_the_data_it_holds(
    send: SendFile, data_uri: str, name: str, content: bytes
):
    status, sent = await send(data_uri)

    assert status == "File sent successfully"
    assert sent == [(name, content)]


@pytest.fixture
def send(mocker, chat: ExecutionContext) -> SendFile:
    """Runs the block in ``chat`` on one file, sending it to a channel that
    records what it got."""

    async def run(file: str, filename: str = "") -> tuple[str, Sent]:
        channel = _RecordingChannel()
        mocker.patch(
            "backend.blocks.discord.bot_blocks.discord.Client",
            side_effect=lambda **kw: _LoggedInClient(channel=channel, **kw),
        )
        block = SendDiscordFileBlock()
        input_data = block.Input.model_validate(
            {
                "channel_identifier": str(_RecordingChannel.id),
                "file": file,
                "filename": filename,
                "credentials": TEST_CREDENTIALS_INPUT,
            }
        )
        outputs = [
            output
            async for output in block.run(
                input_data, credentials=TEST_CREDENTIALS, execution_context=chat
            )
        ]
        return dict(outputs)["status"], channel.sent

    return run


@pytest.fixture
def chat(workspace: UserWorkspace, local_files: None) -> ExecutionContext:
    """The context AutoPilot's run_block gives a block in one of its chats."""
    session_id = str(uuid.uuid4())
    return ExecutionContext(
        user_id=workspace.userId,
        graph_exec_id=f"{COPILOT_SESSION_PREFIX}{session_id}",
        workspace_id=workspace.id,
        session_id=session_id,
    )


@pytest_asyncio.fixture(loop_scope="session")
async def workspace() -> AsyncIterator[UserWorkspace]:
    owned_connection = not db.is_connected()
    if owned_connection:
        await db.connect()
    user_id = str(uuid.uuid4())
    await db.prisma.user.create(
        data={"id": user_id, "email": f"discord-file-{user_id}@example.com"}
    )
    workspace = await db.prisma.userworkspace.create(data={"userId": user_id})
    try:
        yield workspace
    finally:
        await db.prisma.user.delete(where={"id": user_id})
        if owned_connection:
            await db.disconnect()


@pytest.fixture
def local_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Workspace files and the run's local copies kept under ``tmp_path``,
    without the virus scanner, the storage quota or search indexing."""
    monkeypatch.setattr(file_module, "TEMP_DIR", (tmp_path / "temp").resolve())
    storage = LocalWorkspaceStorage(base_dir=str(tmp_path / "storage"))
    with (
        patch(
            "backend.util.workspace.get_workspace_storage",
            AsyncMock(return_value=storage),
        ),
        patch(
            "backend.util.workspace.get_workspace_storage_limit_bytes",
            AsyncMock(return_value=1_000_000),
        ),
        patch("backend.util.workspace.scan_content_safe", AsyncMock()),
        patch("backend.util.file.scan_content_safe", AsyncMock()),
        patch(
            "backend.api.features.workspace.embeddings.schedule_workspace_file_embedding"
        ),
    ):
        yield


async def _write_csv(chat: ExecutionContext) -> str:
    """Saves the CSV in the chat's own folder, as AutoPilot would, and returns
    its file id."""
    assert chat.user_id and chat.workspace_id
    manager = WorkspaceManager(chat.user_id, chat.workspace_id, chat.session_id)
    file = await manager.write_file(content=CSV, filename=FILE_NAME)
    return file.id


class _RecordingChannel:
    """A text channel that keeps each attachment's name and bytes."""

    id = 987654321098765432

    def __init__(self) -> None:
        self.sent: Sent = []

    async def send(
        self, *, content: str | None = None, file: discord.File
    ) -> SimpleNamespace:
        self.sent.append((file.filename, file.fp.read()))
        return SimpleNamespace(id=123456789012345678)


class _LoggedInClient:
    """Stands in for discord.Client: start() logs in at once and runs the
    on_ready handler."""

    def __init__(self, *, channel: _RecordingChannel, **_kwargs) -> None:
        self._channel = channel
        self._on_ready = None
        self.close = AsyncMock()

    def event(self, handler):
        self._on_ready = handler
        return handler

    def get_channel(self, _channel_id: int) -> _RecordingChannel:
        return self._channel

    async def start(self, _token: str) -> None:
        assert self._on_ready is not None
        await self._on_ready()
