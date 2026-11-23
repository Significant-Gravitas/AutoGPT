"""Blocks that stage files of one name in one workflow run each work on their
own file (#17280).

``store_media_file`` names a block's local copy of a file after the file, and
a workflow run keeps every node's copies in its one folder. Two files of one
name used to share a path there: the second copy replaced the first, and a
block joined, read or uploaded the other file under its own file's name.

These run the real blocks on files written through ``WorkspaceManager`` to a
real database, with the real ``store_media_file``. Only the outside world is
replaced: the server an upload goes to, the virus scanner, the storage quota
and search indexing."""

import asyncio
import io
import posixpath
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from moviepy import ColorClip
from openpyxl import Workbook
from prisma.models import UserWorkspace

from backend.blocks._base import Block
from backend.blocks.http import HttpMethod, SendWebRequestBlock
from backend.blocks.spreadsheet import ReadSpreadsheetBlock
from backend.blocks.video._render import open_video
from backend.blocks.video.concat import VideoConcatBlock
from backend.data import db
from backend.data.execution import ExecutionContext
from backend.util import file as file_module
from backend.util.file import parse_workspace_uri
from backend.util.type import MediaFileType
from backend.util.workspace import WorkspaceManager
from backend.util.workspace_storage import LocalWorkspaceStorage

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# What each URL an HTTP Request node posted to got: each file's name and bytes.
Uploads = dict[str, list[tuple[str, bytes]]]


@pytest.mark.asyncio(loop_scope="session")
async def test_video_concat_joins_two_videos_of_one_name(
    graph_run: ExecutionContext, tmp_path: Path
):
    """One Video Concat call given /a/clip.mp4 and /b/clip.mp4 stages both
    before it reads either."""
    red = await _write_file(graph_run, _video(tmp_path, (255, 0, 0)), "/a/clip.mp4")
    blue = await _write_file(graph_run, _video(tmp_path, (0, 0, 255)), "/b/clip.mp4")

    block = VideoConcatBlock()
    outputs = await _run(
        block,
        block.Input(videos=[_ref(red, "video/mp4"), _ref(blue, "video/mp4")]),
        graph_run,
        node_exec_id=str(uuid.uuid4()),
    )

    joined = tmp_path / "joined.mp4"
    joined.write_bytes(await _read_file(graph_run, outputs["video_out"]))
    assert _colours(joined, 0.1, 0.6) == ["red", "blue"]
    assert outputs["total_duration"] == pytest.approx(1.0)


@pytest.mark.asyncio(loop_scope="session")
async def test_two_http_requests_in_one_run_each_upload_their_own_file(
    graph_run: ExecutionContext,
    local_files: AsyncMock,
    uploads: Uploads,
    thread_jobs_wait_their_turn: None,
):
    """Two branches of one run, each with an HTTP Request node that uploads a
    different file called report.csv. The interleaving is forced: see
    ``_second_copy_written_when_the_first_yields``."""
    first = await _write_file(graph_run, b"first report", "/reports/report.csv")
    second = await _write_file(graph_run, b"second report", "/archive/report.csv")
    local_files.side_effect = _second_copy_written_when_the_first_yields(
        first=b"first report"
    )

    # A task group cancels and joins the other node if one fails, so neither
    # outlives the fixtures it runs on.
    async with asyncio.TaskGroup() as nodes:
        for url, file_id in [
            ("https://first.example", first),
            ("https://second.example", second),
        ]:
            nodes.create_task(_upload(graph_run, url, file_id))

    assert uploads == {
        "https://first.example": [("report.csv", b"first report")],
        "https://second.example": [("report.csv", b"second report")],
    }


@pytest.mark.asyncio(loop_scope="session")
async def test_two_spreadsheets_in_one_run_each_read_their_own_rows(
    graph_run: ExecutionContext,
    local_files: AsyncMock,
    thread_jobs_wait_their_turn: None,
):
    """Two Spreadsheet nodes of one run read different workbooks called
    report.xlsx. The block reads a workbook in a worker thread."""
    first_book = _workbook("first")
    first = await _write_file(graph_run, first_book, "/reports/report.xlsx")
    second = await _write_file(graph_run, _workbook("second"), "/archive/report.xlsx")
    local_files.side_effect = _second_copy_written_when_the_first_yields(
        first=first_book
    )

    async with asyncio.TaskGroup() as nodes:
        first_rows = nodes.create_task(_read_rows(graph_run, first))
        second_rows = nodes.create_task(_read_rows(graph_run, second))

    assert first_rows.result() == [{"report": "first"}]
    assert second_rows.result() == [{"report": "second"}]


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
async def workspace() -> AsyncIterator[UserWorkspace]:
    """A user with a workspace in the real database.

    Each undo is registered as soon as what it undoes exists, and every one
    runs even if another fails. Deleting the user takes its workspace and
    files with it."""
    async with AsyncExitStack() as cleanup:
        if not db.is_connected():
            await db.connect()
            cleanup.push_async_callback(db.disconnect)
        user_id = str(uuid.uuid4())
        await db.prisma.user.create(
            data={"id": user_id, "email": f"staged-files-{user_id}@example.com"}
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
            AsyncMock(return_value=10_000_000),
        ),
        patch("backend.util.workspace.scan_content_safe", AsyncMock()),
        patch("backend.util.file.scan_content_safe", AsyncMock()) as scan,
        patch(
            "backend.api.features.workspace.embeddings.schedule_workspace_file_embedding"
        ),
    ):
        yield scan


@pytest.fixture
def uploads() -> Iterator[Uploads]:
    """The server each HTTP Request node posts to, keeping what it got."""
    received: Uploads = {}

    class _Server:
        def __init__(self, **_: Any) -> None:
            pass

        async def request(
            self, method: str, url: str, *, files: list | None = None, **_: Any
        ) -> SimpleNamespace:
            received[url] = [
                (name, data.getvalue()) for _, (name, data, _) in files or []
            ]
            return SimpleNamespace(
                status=200, headers={"content-type": "text/plain"}, text=lambda: "ok"
            )

    with patch("backend.blocks.http.Requests", _Server):
        yield received


@pytest_asyncio.fixture(loop_scope="session")
async def thread_jobs_wait_their_turn() -> AsyncIterator[None]:
    """Runs each job meant for the loop's worker threads on the loop itself,
    behind the callbacks already waiting there, so a read handed to a thread
    happens at a known point instead of whenever a thread gets to it."""
    loop = asyncio.get_running_loop()
    # set_default_executor refuses None, the loop's usual value, so restore it directly
    default_executor = loop._default_executor
    loop.set_default_executor(_InTurnExecutor(loop))
    try:
        yield
    finally:
        loop._default_executor = default_executor


def _second_copy_written_when_the_first_yields(
    *, first: bytes
) -> Callable[..., Awaitable[None]]:
    """A stand-in for the scan, the last step before ``store_media_file``
    writes its copy. The scan of ``first`` returns only once the other node is
    waiting in its own scan, and that one returns right after: the other node
    then writes its copy at the first point the first node yields, which is
    before the first reads its copy."""
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


async def _upload(context: ExecutionContext, url: str, file_id: str) -> None:
    block = SendWebRequestBlock()
    outputs = await _run(
        block,
        block.Input(
            url=url,
            method=HttpMethod.POST,
            json_format=False,
            files=[_ref(file_id, "text/csv")],
        ),
        context,
    )
    assert outputs == {"response": "ok"}


async def _read_rows(context: ExecutionContext, file_id: str) -> list[dict]:
    block = ReadSpreadsheetBlock()
    outputs = await _run(block, block.Input(file_input=_ref(file_id, XLSX)), context)
    return outputs["rows"]


async def _run(
    block: Block, input_data: Any, context: ExecutionContext, **kwargs: Any
) -> dict[str, Any]:
    """Run a block's own ``run()`` as the executor would, and return its
    outputs by name."""
    return {
        name: value
        async for name, value in block.run(
            input_data, execution_context=context, **kwargs
        )
    }


async def _write_file(context: ExecutionContext, content: bytes, path: str) -> str:
    """Save a file in the run's workspace, and return its id."""
    assert context.user_id and context.workspace_id
    manager = WorkspaceManager(context.user_id, context.workspace_id)
    file = await manager.write_file(
        content=content, filename=posixpath.basename(path), path=path
    )
    return file.id


async def _read_file(context: ExecutionContext, reference: str) -> bytes:
    assert context.user_id and context.workspace_id
    manager = WorkspaceManager(context.user_id, context.workspace_id)
    return await manager.read_file_by_id(parse_workspace_uri(reference).file_ref)


def _ref(file_id: str, mime_type: str) -> MediaFileType:
    """The reference a block gets for a workspace file."""
    return MediaFileType(f"workspace://{file_id}#{mime_type}")


def _video(folder: Path, colour: tuple[int, int, int]) -> bytes:
    """Half a second of one colour, the smallest video the video block tests
    use."""
    path = folder / f"{uuid.uuid4().hex}.mp4"
    ColorClip((16, 16), color=colour, duration=0.5).write_videofile(
        str(path), fps=4, logger=None
    )
    return path.read_bytes()


def _colours(video: Path, *seconds: float) -> list[str]:
    """Whether the video is red or blue at each of the given times."""
    clip = open_video(str(video))
    try:
        frames = [clip.get_frame(second) for second in seconds]
    finally:
        clip.close()
    return [
        "red" if frame[..., 0].mean() > frame[..., 2].mean() else "blue"
        for frame in frames
    ]


def _workbook(value: str) -> bytes:
    """A workbook with one column, ``report``, holding *value*."""
    book = Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.append(["report"])
    sheet.append([value])
    content = io.BytesIO()
    book.save(content)
    return content.getvalue()
