from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.integrations.webhooks.graph_lifecycle_hooks import (
    GraphActivationError,
    _before_graph_activate,
    on_graph_deactivate,
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


@pytest.mark.asyncio
async def test_before_graph_activate_ignores_credential_meta_without_id():
    """A credentials value that is truthy but carries no `id` means "nothing
    selected", not "resolve this". It must be skipped like an absent field
    rather than indexed — indexing raised KeyError('id'), surfacing to the user
    as a bare 500 on POST /api/graphs that named neither block nor field."""
    node = _make_node(creds_field="codex_credentials", required=False)
    node.input_default = {"codex_credentials": {"provider": "codex", "type": "oauth2"}}
    graph = MagicMock(nodes=[node])

    getter = AsyncMock(return_value=MagicMock())
    with patch(
        "backend.integrations.webhooks.graph_lifecycle_hooks.credentials_manager"
    ) as mgr:
        mgr.cached_getter.return_value = getter
        await _before_graph_activate(graph, "user-1")

    getter.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_graph_deactivate_ignores_credential_meta_without_id():
    """Deactivation had the same `creds_meta["id"]` indexing as activation.
    Graphs persisted with the id-less shape hit it on delete, so the guard
    needs its own regression test rather than relying on the activation one."""
    node = _make_node(creds_field="codex_credentials", required=False)
    node.input_default = {"codex_credentials": {"provider": "codex", "type": "oauth2"}}
    node.block.webhook_config = None
    graph = MagicMock(nodes=[node])

    getter = AsyncMock(return_value=MagicMock())
    with patch(
        "backend.integrations.webhooks.graph_lifecycle_hooks.credentials_manager"
    ) as mgr:
        mgr.cached_getter.return_value = getter
        await on_graph_deactivate(graph, "user-1")

    getter.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_graph_deactivate_keeps_first_resolved_credential():
    """A failed lookup on a later field must not discard a credential an
    earlier field already resolved."""
    node = _make_node(required=False)
    node.input_default = {
        "credentials": {"id": "good", "provider": "github", "type": "api_key"},
        "other_credentials": {"id": "gone", "provider": "github", "type": "api_key"},
    }
    node.block.webhook_config = None
    schema = node.block.input_schema
    schema.get_credentials_fields.return_value = {
        "credentials": object(),
        "other_credentials": object(),
    }

    resolved = MagicMock()

    async def getter(creds_id):
        return resolved if creds_id == "good" else None

    graph = MagicMock(nodes=[node])
    with patch(
        "backend.integrations.webhooks.graph_lifecycle_hooks.credentials_manager"
    ) as mgr:
        mgr.cached_getter.return_value = getter
        with patch(
            "backend.integrations.webhooks.graph_lifecycle_hooks.on_node_deactivate",
            new=AsyncMock(return_value=node),
        ) as deactivate:
            await on_graph_deactivate(graph, "user-1")

    assert deactivate.await_args.kwargs["credentials"] is resolved
