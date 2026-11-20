"""#16234: a node whose API key has expired fails as the user's error instead of
running, and a dry run, which never uses the key, still simulates it."""

import time
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks._base import BlockSchemaInput
from backend.blocks.github.issues import GithubCommentBlock
from backend.data.execution import ExecutionContext, ExecutionStatus, NodeExecutionEntry
from backend.data.graph import Node
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    NodeExecutionStats,
)
from backend.executor.manager import ExecutionProcessor, execute_node
from backend.executor.utils import NodeExecutionProgress
from backend.integrations.providers import ProviderName
from backend.util.exceptions import CredentialsExpiredError

INPUTS = {
    "credentials": {"id": "cred-1", "provider": "github", "type": "api_key"},
    "issue_url": "https://github.com/o/r/issues/1",
    "comment": "hi",
}


class _DefaultedKeyInput(BlockSchemaInput):
    credentials: CredentialsMetaInput[
        Literal[ProviderName.GITHUB], Literal["api_key"]
    ] = CredentialsField(description="GitHub key", default={})


def _defaulted_key_block() -> GithubCommentBlock:
    """One instance whose key field has a default, which the executor runs
    without a key it can't load. (A Block subclass would join every block
    list in the process.)"""
    block = GithubCommentBlock()
    block.input_schema = _DefaultedKeyInput
    return block


def _expired_key() -> APIKeyCredentials:
    return APIKeyCredentials(
        id="cred-1",
        provider="github",
        api_key=SecretStr("k"),
        title="GH",
        expires_at=int(time.time()) - 60,
    )


def _processor(key: APIKeyCredentials) -> ExecutionProcessor:
    processor = ExecutionProcessor()
    processor.creds_manager = MagicMock(get=AsyncMock(return_value=key))
    return processor


def _node() -> Node:
    return Node(id="node-1", block_id=GithubCommentBlock().id, input_default=INPUTS)


def _node_exec(
    dry_run: bool = False, block_id: str = "", inputs: dict = INPUTS
) -> NodeExecutionEntry:
    return NodeExecutionEntry(
        user_id="user-1",
        graph_exec_id="graph-exec-1",
        graph_id="graph-1",
        graph_version=1,
        node_exec_id="node-exec-1",
        node_id="node-1",
        block_id=block_id or GithubCommentBlock().id,
        inputs=dict(inputs),
        execution_context=ExecutionContext(
            root_execution_id="graph-exec-1",
            dry_run=dry_run,
            user_timezone="America/Chicago",
        ),
    )


@pytest.mark.parametrize(
    "block, inputs",
    [
        (GithubCommentBlock(), INPUTS),
        # Without a key it can't load, a field with a default runs keyless;
        # an expired key must not take that way out.
        (_defaulted_key_block(), {"credentials": INPUTS["credentials"]}),
    ],
    ids=["required-field", "field-with-default"],
)
@pytest.mark.asyncio
async def test_an_expired_api_key_fails_the_node_as_the_users_error(
    block: GithubCommentBlock, inputs: dict
):
    key = _expired_key()
    run = AsyncMock()
    stats = NodeExecutionStats()
    log_metadata = MagicMock()
    db_client = MagicMock(
        upsert_execution_output=AsyncMock(),
        get_node_execution=AsyncMock(return_value=None),
    )
    with (
        patch("backend.data.graph.get_block", return_value=block),
        patch("backend.executor.utils.get_block", return_value=block),
        patch.object(type(block), "run", run),
        patch(
            "backend.executor.manager.async_update_node_execution_status",
            new=AsyncMock(),
        ),
    ):
        _, status = await _processor(key)._on_node_execution(
            node=Node(id="node-1", block_id=block.id, input_default=inputs),
            node_exec=_node_exec(block_id=block.id, inputs=inputs),
            node_exec_progress=NodeExecutionProgress(),
            stats=stats,
            db_client=db_client,
            log_metadata=log_metadata,
        )

    assert status == ExecutionStatus.FAILED
    assert isinstance(stats.error, CredentialsExpiredError)
    assert "API key 'GH' expired on" in str(stats.error)
    # Dated in the user's own zone, so it names the day they picked.
    assert str(stats.error) == str(key.expired_error("America/Chicago"))
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
                _node(), _node_exec(dry_run=True), _processor(_expired_key())
            )
        ]

    assert outputs == [("comment_id", "simulated")]
    run.assert_not_called()
