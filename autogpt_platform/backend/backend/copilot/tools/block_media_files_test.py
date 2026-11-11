"""A block run from AutoPilot keeps the media files it stores locally only for
as long as the call runs, without touching other calls from the same chat."""

import asyncio
import base64
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.blocks._base import BlockType
from backend.copilot.tools.helpers import execute_block
from backend.copilot.tools.models import BlockOutputResponse, ErrorResponse
from backend.data.execution import ExecutionContext
from backend.util import file as file_module
from backend.util.file import get_exec_file_path, store_media_file
from backend.util.type import MediaFileType

_SESSION = "media-files-session"
BlockRun = Callable[[ExecutionContext], Awaitable[str]]


@pytest.fixture(autouse=True)
def isolated_media(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(file_module, "TEMP_DIR", tmp_path)
    monkeypatch.setattr(file_module, "scan_content_safe", AsyncMock())
    storage = MagicMock()
    storage.is_cloud_path.return_value = False
    monkeypatch.setattr(
        file_module, "get_cloud_storage_handler", AsyncMock(return_value=storage)
    )
    workspace_client = MagicMock()
    workspace_client.get_or_create_workspace = AsyncMock(
        return_value=MagicMock(id="ws-1")
    )
    monkeypatch.setattr(
        "backend.copilot.tools.helpers.workspace_db", lambda: workspace_client
    )
    monkeypatch.setattr(
        "backend.copilot.tools.helpers.block_usage_cost", lambda *_, **__: (0, {})
    )
    return tmp_path


def _media_block(run: BlockRun) -> MagicMock:
    block = MagicMock()
    block.id = "media-block"
    block.name = "MediaBlock"
    block.block_type = BlockType.STANDARD
    block.input_schema.get_credentials_fields_info.return_value = {}
    block.input_schema.validate_data.return_value = None

    async def execute(
        input_data: dict, **kwargs: Any
    ) -> AsyncIterator[tuple[str, Any]]:
        yield "result", await run(kwargs["execution_context"])

    block.execute = execute
    return block


async def _store(context: ExecutionContext, text: str) -> Path:
    """Store a file the way a media block does, and return where it landed."""
    assert context.graph_exec_id is not None
    encoded = base64.b64encode(text.encode()).decode()
    local = await store_media_file(
        file=MediaFileType(f"data:text/plain;base64,{encoded}"),
        execution_context=context,
        return_format="for_local_processing",
    )
    return Path(get_exec_file_path(context.graph_exec_id, local))


async def _call(run: BlockRun, node_exec_id: str = "exec-1") -> Any:
    return await execute_block(
        block=_media_block(run),
        block_id="media-block",
        input_data={},
        user_id="media-files-user",
        session_id=_SESSION,
        node_exec_id=node_exec_id,
        matched_credentials={},
        dry_run=False,
    )


def _leftovers(temp_dir: Path) -> list[Path]:
    return list((temp_dir / "exec_file").rglob("*"))


async def test_a_block_s_media_files_are_gone_once_the_call_returns(
    isolated_media: Path,
) -> None:
    stored: list[Path] = []

    async def run(context: ExecutionContext) -> str:
        path = await _store(context, "frame")
        stored.append(path)
        return path.read_text()

    result = await _call(run)

    assert isinstance(result, BlockOutputResponse)
    assert result.outputs == {"result": ["frame"]}
    assert len(stored) == 1 and not stored[0].exists()
    assert _leftovers(isolated_media) == []


async def test_a_failed_call_s_media_files_are_gone_too(
    isolated_media: Path,
) -> None:
    stored: list[Path] = []

    async def run(context: ExecutionContext) -> str:
        stored.append(await _store(context, "frame"))
        raise RuntimeError("encoder crashed")

    result = await _call(run)

    assert isinstance(result, ErrorResponse)
    assert len(stored) == 1 and not stored[0].exists()
    assert _leftovers(isolated_media) == []


async def test_a_cancelled_call_s_media_files_are_gone_too(
    isolated_media: Path,
) -> None:
    """The user stopped the turn, or the call ran past the wait cap."""
    stored: list[Path] = []
    file_stored = asyncio.Event()

    async def run(context: ExecutionContext) -> str:
        stored.append(await _store(context, "frame"))
        file_stored.set()
        await asyncio.Event().wait()
        return "never"

    call = asyncio.create_task(_call(run))
    await asyncio.wait_for(file_stored.wait(), timeout=30)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    assert len(stored) == 1 and not stored[0].exists()
    assert _leftovers(isolated_media) == []


async def test_a_call_that_ends_leaves_a_running_sibling_s_files_alone(
    isolated_media: Path,
) -> None:
    """Calls from one assistant message run at the same time, in one chat."""
    first_stored = asyncio.Event()
    second_returned = asyncio.Event()
    stored: dict[str, Path] = {}

    async def first(context: ExecutionContext) -> str:
        stored["first"] = await _store(context, "first frame")
        first_stored.set()
        await second_returned.wait()
        return stored["first"].read_text()

    async def second(context: ExecutionContext) -> str:
        stored["second"] = await _store(context, "second frame")
        return "done"

    first_call = asyncio.create_task(_call(first, node_exec_id="exec-first"))
    await asyncio.wait_for(first_stored.wait(), timeout=30)
    second_result = await _call(second, node_exec_id="exec-second")

    assert isinstance(second_result, BlockOutputResponse)
    assert not stored["second"].exists()
    assert stored["first"].exists()

    second_returned.set()
    first_result = await asyncio.wait_for(first_call, timeout=30)

    assert isinstance(first_result, BlockOutputResponse)
    assert first_result.outputs == {"result": ["first frame"]}
    assert _leftovers(isolated_media) == []
