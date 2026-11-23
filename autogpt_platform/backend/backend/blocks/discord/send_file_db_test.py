"""What Send Discord File attaches, and under which name (#17272).

The block's own ``run()`` with the context AutoPilot gives it, a file written
through ``WorkspaceManager`` to a real database and the real
``store_media_file``. Only the Discord client is replaced, by one that records
what the channel was sent."""

import asyncio
import base64
import posixpath
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch
from urllib.parse import quote

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
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
GREETING = "Grüße aus München".encode()

# Each attachment the channel got, as (name, bytes).
Sent = list[tuple[str, bytes]]
# Runs the block in a context on a file (and an optional filename): its
# status, and what was sent.
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
    file_id = await _write_file(chat)

    status, sent = await send(chat, reference.format(id=file_id, name=FILE_NAME))

    assert status == "File sent successfully"
    assert sent == [(FILE_NAME, CSV)]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_filename_given_to_the_block_wins(
    send: SendFile, chat: ExecutionContext
):
    file_id = await _write_file(chat)

    status, sent = await send(chat, f"workspace://{file_id}#text/csv", "Bericht.csv")

    assert status == "File sent successfully"
    assert sent == [("Bericht.csv", CSV)]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_run_file_goes_out_under_its_own_name(
    send: SendFile, graph_run: ExecutionContext
):
    """A file an earlier node left in the workflow run's own folder."""
    assert graph_run.graph_exec_id
    path = Path(file_module.get_exec_file_path(graph_run.graph_exec_id, "image.png"))
    path.parent.mkdir(parents=True)
    path.write_bytes(PNG)

    status, sent = await send(graph_run, "image.png")

    assert status == "File sent successfully"
    assert sent == [("image.png", PNG)]


@pytest.mark.asyncio(loop_scope="session")
async def test_two_sends_in_one_run_each_carry_their_own_file(
    send: SendFile,
    graph_run: ExecutionContext,
    local_files: AsyncMock,
    thread_jobs_wait_their_turn: None,
):
    """Two nodes of one workflow run send different workspace files that have
    the same name. The run keeps every node's copies in one folder, so the
    second copy replaces the first: each send has to read its own copy before
    another node can write there.

    The interleaving is forced. The first send writes its copy only once the
    second is waiting right before writing its own, so the second writes at
    the first point the first send yields. A job meant for a worker thread
    runs on the loop in turn, so a read moved to a thread comes after that
    write as well."""
    first = await _write_file(graph_run, b"first report", "/reports/report.csv")
    second = await _write_file(graph_run, b"second report", "/archive/report.csv")
    local_files.side_effect = _second_copy_written_when_the_first_yields(
        first=b"first report"
    )

    # A task group cancels and joins the other send if one fails, so neither
    # outlives the fixtures it runs on.
    async with asyncio.TaskGroup() as sends:
        first_send = sends.create_task(send(graph_run, f"workspace://{first}#text/csv"))
        second_send = sends.create_task(
            send(graph_run, f"workspace://{second}#text/csv")
        )

    (first_status, first_sent), (second_status, second_sent) = (
        first_send.result(),
        second_send.result(),
    )
    assert [first_status, second_status] == ["File sent successfully"] * 2
    assert first_sent == [("report.csv", b"first report")]
    assert second_sent == [("report.csv", b"second report")]


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "data_uri, name, content",
    [
        (
            f"data:application/pdf;base64,{base64.b64encode(PDF).decode()}",
            "file.pdf",
            PDF,
        ),
        (
            "data:text/plain;charset=utf-8;base64,"
            + base64.b64encode(GREETING).decode(),
            "file.txt",
            GREETING,
        ),
        ("data:text/plain,Hallo%20Welt", "file.txt", b"Hallo Welt"),
        (
            "data:text/plain;charset=utf-8,Gr%C3%BC%C3%9Fe%20aus%20M%C3%BCnchen",
            "file.txt",
            GREETING,
        ),
        # RFC 2397: a data URI that names no media type holds text/plain.
        ("data:,Hallo", "file.txt", b"Hallo"),
    ],
    ids=[
        "base64",
        "base64-with-charset",
        "percent-encoded",
        "percent-encoded-utf8",
        "no-media-type",
    ],
)
async def test_a_data_uri_goes_out_as_the_data_it_holds(
    send: SendFile, chat: ExecutionContext, data_uri: str, name: str, content: bytes
):
    status, sent = await send(chat, data_uri)

    assert status == "File sent successfully"
    assert sent == [(name, content)]


@pytest.mark.asyncio(loop_scope="session")
async def test_escaping_a_base64_data_uri_sends_the_same_bytes(
    send: SendFile, chat: ExecutionContext
):
    """RFC 2397 data is percent-encoded, base64 or not, so escaping the
    base64's ``+``, ``/`` and ``=`` must not change what is sent."""
    data = bytes([0xFB, 0xFF, 0xBF, 0x01])  # base64: +/+/AQ==
    payload = base64.b64encode(data).decode()

    plain = await send(chat, f"data:application/octet-stream;base64,{payload}")
    escaped = await send(
        chat, f"data:application/octet-stream;base64,{quote(payload, safe='')}"
    )

    assert plain == ("File sent successfully", [("file.bin", data)])
    assert escaped == plain


@pytest.fixture
def send(mocker) -> SendFile:
    """Runs the block in a context on one file, sending it to a channel that
    records what it got."""

    async def run(
        context: ExecutionContext, file: str, filename: str = ""
    ) -> tuple[str, Sent]:
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
                input_data, credentials=TEST_CREDENTIALS, execution_context=context
            )
        ]
        return dict(outputs)["status"], channel.sent

    return run


@pytest.fixture
def chat(workspace: UserWorkspace, local_files: AsyncMock) -> ExecutionContext:
    """The context AutoPilot's run_block gives a block in one of its chats."""
    session_id = str(uuid.uuid4())
    return ExecutionContext(
        user_id=workspace.userId,
        graph_exec_id=f"{COPILOT_SESSION_PREFIX}{session_id}",
        workspace_id=workspace.id,
        session_id=session_id,
    )


@pytest.fixture
def graph_run(workspace: UserWorkspace, local_files: AsyncMock) -> ExecutionContext:
    """The context the executor gives a block in a workflow run outside
    AutoPilot: the owner's workspace and no chat, so every node of the run
    keeps its copies in the run's one folder."""
    return ExecutionContext(
        user_id=workspace.userId,
        graph_id=str(uuid.uuid4()),
        graph_exec_id=str(uuid.uuid4()),
        workspace_id=workspace.id,
    )


@pytest_asyncio.fixture(loop_scope="session")
async def thread_jobs_wait_their_turn() -> AsyncIterator[None]:
    """Runs each job meant for the loop's worker threads on the loop itself,
    behind the callbacks already waiting there, so a read handed to a thread
    happens at a known point instead of whenever a thread gets to it."""
    loop = asyncio.get_running_loop()
    default_executor = loop._default_executor
    loop.set_default_executor(_InTurnExecutor(loop))
    try:
        yield
    finally:
        loop._default_executor = default_executor


@pytest_asyncio.fixture(loop_scope="session")
async def workspace() -> AsyncIterator[UserWorkspace]:
    """A user with a workspace in the real database.

    Each undo is registered as soon as what it undoes exists, and every one
    runs even if another fails: a failed step leaves no row or connection
    behind. Deleting the user takes its workspace and files with it."""
    async with AsyncExitStack() as cleanup:
        if not db.is_connected():
            await db.connect()
            cleanup.push_async_callback(db.disconnect)
        user_id = str(uuid.uuid4())
        await db.prisma.user.create(
            data={"id": user_id, "email": f"discord-file-{user_id}@example.com"}
        )
        cleanup.push_async_callback(db.prisma.user.delete_many, where={"id": user_id})
        yield await db.prisma.userworkspace.create(data={"userId": user_id})


@pytest.fixture
def local_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AsyncMock]:
    """Workspace files and the run's local copies kept under ``tmp_path``,
    without the virus scanner, the storage quota or search indexing.

    Yields the stand-in for the scan ``store_media_file`` runs right before it
    writes its copy."""
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
        patch("backend.util.file.scan_content_safe", AsyncMock()) as scan,
        patch(
            "backend.api.features.workspace.embeddings.schedule_workspace_file_embedding"
        ),
    ):
        yield scan


async def _write_file(
    context: ExecutionContext, content: bytes = CSV, path: str = f"/{FILE_NAME}"
) -> str:
    """Saves a file in the context's workspace, inside its chat's own folder
    when it has a chat, as AutoPilot would, and returns its file id."""
    assert context.user_id and context.workspace_id
    manager = WorkspaceManager(
        context.user_id, context.workspace_id, context.session_id
    )
    file = await manager.write_file(
        content=content, filename=posixpath.basename(path), path=path
    )
    return file.id


def _second_copy_written_when_the_first_yields(
    *, first: bytes
) -> Callable[..., Awaitable[None]]:
    """A stand-in for the scan, the last step before ``store_media_file``
    writes its copy. The scan of ``first`` returns only once the other send is
    waiting in its own scan, and that one returns right after: the other send
    then writes its copy at the first point the first send yields."""
    second_waiting = asyncio.Event()
    first_scanned = asyncio.Event()

    async def scan(content: bytes, *, filename: str) -> None:
        async with asyncio.timeout(30):
            if content == first:
                await second_waiting.wait()
                first_scanned.set()
            else:
                second_waiting.set()
                await first_scanned.wait()

    return scan


class _InTurnExecutor(ThreadPoolExecutor):
    """Queues each job on the event loop, behind the callbacks already waiting
    there, instead of handing it to a thread."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__(max_workers=1)
        self._loop = loop

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:
        job: Future = Future()

        def run() -> None:
            if not job.set_running_or_notify_cancel():
                return
            try:
                job.set_result(fn(*args, **kwargs))
            except BaseException as error:
                job.set_exception(error)

        self._loop.call_soon(run)
        return job


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
