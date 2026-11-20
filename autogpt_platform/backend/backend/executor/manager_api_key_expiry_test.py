"""#16234: a node whose API key has expired fails as the user's error instead of
running, and a dry run, which never uses the key, still simulates it."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.github.issues import GithubCommentBlock
from backend.data.execution import ExecutionContext, ExecutionStatus, NodeExecutionEntry
from backend.data.graph import Node
from backend.data.model import APIKeyCredentials, NodeExecutionStats
from backend.executor.manager import ExecutionProcessor, execute_node
from backend.executor.utils import NodeExecutionProgress
from backend.util.exceptions import CredentialsExpiredError

INPUTS = {
    "credentials": {"id": "cred-1", "provider": "github", "type": "api_key"},
    "issue_url": "https://github.com/o/r/issues/1",
    "comment": "hi",
}


def _processor() -> ExecutionProcessor:
    processor = ExecutionProcessor()
    expired = APIKeyCredentials(
        id="cred-1",
        provider="github",
        api_key=SecretStr("k"),
        title="GH",
        expires_at=int(time.time()) - 60,
    )
    processor.creds_manager = MagicMock(get=AsyncMock(return_value=expired))
    return processor


def _node() -> Node:
    return Node(id="node-1", block_id=GithubCommentBlock().id, input_default=INPUTS)


def _node_exec(dry_run: bool = False) -> NodeExecutionEntry:
    return NodeExecutionEntry(
        user_id="user-1",
        graph_exec_id="graph-exec-1",
        graph_id="graph-1",
        graph_version=1,
        node_exec_id="node-exec-1",
        node_id="node-1",
        block_id=GithubCommentBlock().id,
        inputs=dict(INPUTS),
        execution_context=ExecutionContext(
            root_execution_id="graph-exec-1", dry_run=dry_run
        ),
    )


@pytest.mark.asyncio
async def test_an_expired_api_key_fails_the_node_as_the_users_error():
    run = AsyncMock()
    stats = NodeExecutionStats()
    log_metadata = MagicMock()
    db_client = MagicMock(
        upsert_execution_output=AsyncMock(),
        get_node_execution=AsyncMock(return_value=None),
    )
    with (
        patch.object(GithubCommentBlock, "run", run),
        patch(
            "backend.executor.manager.async_update_node_execution_status",
            new=AsyncMock(),
        ),
    ):
        _, status = await _processor()._on_node_execution(
            node=_node(),
            node_exec=_node_exec(),
            node_exec_progress=NodeExecutionProgress(),
            stats=stats,
            db_client=db_client,
            log_metadata=log_metadata,
        )

    assert status == ExecutionStatus.FAILED
    assert isinstance(stats.error, CredentialsExpiredError)
    assert "API key 'GH' expired on" in str(stats.error)
    run.assert_not_called()
    # An expected failure: logged as an error, it would reach Sentry on every
    # scheduled run until the user adds a new key.
    log_metadata.exception.assert_not_called()


@pytest.mark.asyncio
async def test_a_dry_run_still_simulates_a_node_whose_key_expired():
    async def simulate(block, input_data, user_id):
        yield "comment_id", "simulated"

    run = AsyncMock()
    with (
        patch.object(GithubCommentBlock, "run", run),
        patch("backend.executor.manager.simulate_block", new=simulate),
        patch(
            "backend.executor.manager._sentry_get_current_scope",
            return_value=MagicMock(_user=None, _tags={}),
        ),
    ):
        outputs = [
            output
            async for output in execute_node(
                _node(), _node_exec(dry_run=True), _processor()
            )
        ]

    assert outputs == [("comment_id", "simulated")]
    run.assert_not_called()
