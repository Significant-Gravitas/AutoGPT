"""The local copies a node makes of its input files go when the node ends,
and only its own: a node of the same run that is still running keeps its
copies (#17280)."""

import asyncio
import base64
import logging
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    NodeExecutionEntry,
)
from backend.data.graph import Node
from backend.data.model import NodeExecutionStats
from backend.executor.manager import ExecutionProcessor
from backend.executor.utils import LogMetadata, NodeExecutionProgress
from backend.util import file as file_module
from backend.util.file import get_exec_file_path, store_media_file
from backend.util.type import MediaFileType

_GRAPH_EXEC_ID = str(uuid.uuid4())


@pytest.fixture(autouse=True)
def exec_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(file_module, "TEMP_DIR", tmp_path.resolve())
    monkeypatch.setattr(file_module, "scan_content_safe", AsyncMock())
    storage = MagicMock()
    storage.is_cloud_path.return_value = False
    monkeypatch.setattr(
        file_module, "get_cloud_storage_handler", AsyncMock(return_value=storage)
    )
    return tmp_path.resolve() / "exec_file"


async def test_a_node_s_copies_go_when_it_ends_and_not_before(
    exec_files: Path,
) -> None:
    first_staged = asyncio.Event()
    second_ended = asyncio.Event()
    copies: dict[str, Path] = {}

    async def run_block(
        *, node: Node, data: NodeExecutionEntry, **_: Any
    ) -> AsyncIterator[tuple[str, str]]:
        copies[node.id] = await _stage(data, f"{node.id} node's file")
        if node.id == "first":
            first_staged.set()
            await second_ended.wait()
        yield "result", copies[node.id].read_text()

    with (
        patch("backend.executor.manager.execute_node", new=run_block),
        patch(
            "backend.executor.manager.async_update_node_execution_status",
            new=AsyncMock(),
        ),
        patch(
            "backend.data.graph.get_block",
            return_value=SimpleNamespace(execution_timeout_seconds=60),
        ),
    ):
        processor = ExecutionProcessor()
        first = asyncio.ensure_future(_run_node(processor, "first"))
        try:
            await asyncio.wait_for(first_staged.wait(), timeout=30)
            second = await asyncio.wait_for(_run_node(processor, "second"), 30)

            assert second == (ExecutionStatus.COMPLETED, ["second node's file"])
            assert not copies["second"].parent.exists()
            assert copies["first"].read_text() == "first node's file"

            second_ended.set()
            assert await asyncio.wait_for(first, timeout=30) == (
                ExecutionStatus.COMPLETED,
                ["first node's file"],
            )
        finally:
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)

    assert not copies["first"].parent.exists()
    assert list((exec_files / _GRAPH_EXEC_ID).iterdir()) == []


async def _run_node(
    processor: ExecutionProcessor, node_id: str
) -> tuple[ExecutionStatus, list[Any]]:
    """Run a node of the run as the executor does, and return its status
    and the data it output."""
    progress = NodeExecutionProgress()
    node_exec = NodeExecutionEntry(
        user_id="user-1",
        graph_exec_id=_GRAPH_EXEC_ID,
        graph_id="graph-1",
        graph_version=1,
        node_exec_id=f"{node_id}-exec",
        node_id=node_id,
        block_id="block-1",
        inputs={},
        execution_context=ExecutionContext(
            user_id="user-1", graph_exec_id=_GRAPH_EXEC_ID
        ),
    )
    _, status = await processor._on_node_execution(
        node=Node(id=node_id, block_id="block-1"),
        node_exec=node_exec,
        node_exec_progress=progress,
        stats=NodeExecutionStats(),
        db_client=MagicMock(
            upsert_execution_output=AsyncMock(),
            get_node_execution=AsyncMock(return_value=None),
        ),
        log_metadata=LogMetadata(
            logger=logging.getLogger(__name__),
            user_id="user-1",
            graph_eid=_GRAPH_EXEC_ID,
            graph_id="graph-1",
            node_eid=node_exec.node_exec_id,
            node_id=node_id,
            block_name="MediaBlock",
        ),
    )
    outputs = progress.output.get(node_exec.node_exec_id, [])
    return status, [output.data[1] for output in outputs]


async def _stage(node_exec: NodeExecutionEntry, content: str) -> Path:
    """Stage a file for local processing as a block does, and return where
    its copy is."""
    encoded = base64.b64encode(content.encode()).decode()
    local_path = await store_media_file(
        MediaFileType(f"data:text/plain;base64,{encoded}"),
        ExecutionContext(user_id="user-1", graph_exec_id=node_exec.graph_exec_id),
        return_format="for_local_processing",
    )
    return Path(get_exec_file_path(node_exec.graph_exec_id, local_path))
