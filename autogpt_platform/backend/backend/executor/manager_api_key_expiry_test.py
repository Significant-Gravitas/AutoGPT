"""#16234: a node whose API key has expired fails instead of running."""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.github.issues import GithubCommentBlock
from backend.data.execution import NodeExecutionEntry
from backend.data.graph import Node
from backend.data.model import APIKeyCredentials
from backend.executor.manager import execute_node
from backend.util.exceptions import CredentialsExpiredError

META = {"id": "cred-1", "provider": "github", "type": "api_key", "title": "GH"}


@pytest.mark.asyncio
async def test_expired_api_key_fails_the_node():
    """The field is required (its default is not None), so the `except
    ValueError` fallback in execute_node would run it without credentials
    if the expiry were raised as a ValueError."""
    block = GithubCommentBlock()
    assert block.input_schema.model_fields["credentials"].default is not None
    inputs = {
        "credentials": META,
        "issue_url": "https://github.com/o/r/issues/1",
        "comment": "hi",
    }
    node = Node(id="node-1", block_id=block.id, input_default=inputs)
    entry = NodeExecutionEntry(
        user_id="user-1",
        graph_exec_id="graph-exec-1",
        graph_id="graph-1",
        graph_version=1,
        node_exec_id="node-exec-1",
        node_id=node.id,
        block_id=block.id,
        inputs=inputs,
    )
    creds_manager = MagicMock()
    creds_manager.get = AsyncMock(
        return_value=APIKeyCredentials(
            id="cred-1",
            provider="github",
            api_key=SecretStr("k"),
            title="GH",
            expires_at=int(time.time()) - 60,
        )
    )
    run = AsyncMock()
    scope = MagicMock(_user=None, _tags={})
    with (
        patch.object(GithubCommentBlock, "run", run),
        patch("backend.executor.manager._sentry_get_current_scope", return_value=scope),
        patch("backend.executor.manager._sentry_capture_exception") as capture,
        pytest.raises(CredentialsExpiredError, match="API key 'GH' expired on"),
    ):
        async for _ in execute_node(
            node, entry, SimpleNamespace(creds_manager=creds_manager)
        ):
            pass

    run.assert_not_called()
    # A user's own expired key is not a platform error.
    capture.assert_not_called()
