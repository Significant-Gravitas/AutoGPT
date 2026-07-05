from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.integrations.webhooks.graph_lifecycle_hooks import (
    GraphActivationError,
    _before_graph_activate,
)


def _make_node(
    *,
    creds_field: str = "credentials",
    creds_id: str = "cred-1",
    creds_title: str | None = "My GitHub key",
    creds_provider: str = "github",
    block_name: str = "GithubCommentBlock",
    required: bool = True,
    optional_marker: bool = False,
):
    block_input_schema = MagicMock()
    block_input_schema.get_credentials_fields.return_value = {creds_field: object()}
    block_input_schema.get_required_fields.return_value = (
        {creds_field} if required else set()
    )

    node = MagicMock()
    node.id = "node-1"
    node.credentials_optional = optional_marker
    cred = {"id": creds_id, "provider": creds_provider}
    if creds_title is not None:
        cred["title"] = creds_title
    node.input_default = {creds_field: cred}
    node.block.input_schema = block_input_schema
    node.block.name = block_name
    return node


@pytest.mark.asyncio
async def test_before_graph_activate_oauth_refresh_failure_raises_clear_error():
    """A required credential whose OAuth refresh raises (e.g. invalid_grant)
    must surface as GraphActivationError with a 'please reconnect' message —
    not as an opaque 500."""
    node = _make_node()
    graph = MagicMock(nodes=[node])

    async def failing_getter(_creds_id):
        raise Exception("invalid_grant: Bad Request")

    with patch(
        "backend.integrations.webhooks.graph_lifecycle_hooks.credentials_manager"
    ) as mgr:
        mgr.cached_getter.return_value = failing_getter
        with pytest.raises(GraphActivationError) as excinfo:
            await _before_graph_activate(graph, "user-1")

    msg = str(excinfo.value)
    # User-facing reference must use the credential title + provider + block
    # name (the things a user can act on), not internal UUIDs.
    assert "My GitHub key" in msg
    assert "github" in msg.lower()
    assert "GithubCommentBlock" in msg
    assert "cred-1" not in msg
    assert "node-1" not in msg
    assert "reconnect" in msg.lower()
    assert "invalid_grant" in msg


@pytest.mark.asyncio
async def test_before_graph_activate_clears_optional_unloadable_credentials():
    """An optional credential whose refresh fails should be cleared, not raise."""
    node = _make_node(required=False, optional_marker=True)
    graph = MagicMock(nodes=[node])

    async def failing_getter(_creds_id):
        raise Exception("invalid_grant: Bad Request")

    with patch(
        "backend.integrations.webhooks.graph_lifecycle_hooks.credentials_manager"
    ) as mgr:
        mgr.cached_getter.return_value = failing_getter
        await _before_graph_activate(graph, "user-1")

    assert node.input_default["credentials"] == {}


@pytest.mark.asyncio
async def test_before_graph_activate_missing_required_credential_raises_clear_error():
    """A required credential that no longer exists in the DB (returns None)
    raises GraphActivationError that names the missing credential and asks
    the user to pick a different one."""
    node = _make_node()
    graph = MagicMock(nodes=[node])

    with patch(
        "backend.integrations.webhooks.graph_lifecycle_hooks.credentials_manager"
    ) as mgr:
        mgr.cached_getter.return_value = AsyncMock(return_value=None)
        with pytest.raises(GraphActivationError) as excinfo:
            await _before_graph_activate(graph, "user-1")

    msg = str(excinfo.value)
    assert "My GitHub key" in msg
    assert "github" in msg.lower()
    assert "GithubCommentBlock" in msg
    assert "cred-1" not in msg
    assert "no longer exists" in msg.lower()


@pytest.mark.asyncio
async def test_before_graph_activate_succeeds_when_credentials_resolve():
    """The happy path should be a no-op (no mutation, no raise)."""
    node = _make_node()
    graph = MagicMock(nodes=[node])

    with patch(
        "backend.integrations.webhooks.graph_lifecycle_hooks.credentials_manager"
    ) as mgr:
        mgr.cached_getter.return_value = AsyncMock(return_value=MagicMock())
        await _before_graph_activate(graph, "user-1")

    assert node.input_default["credentials"]["id"] == "cred-1"
