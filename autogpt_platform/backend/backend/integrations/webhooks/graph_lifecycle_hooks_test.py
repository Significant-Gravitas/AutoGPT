from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.autopilot import AutoPilotBlock, AutoPilotTransport
from backend.blocks.basic import StoreValueBlock
from backend.blocks.google.sheets import GoogleSheetsReadBlock
from backend.data.execution import ExecutionContext
from backend.data.graph import Graph, Node, make_graph_model
from backend.integrations.webhooks.graph_lifecycle_hooks import (
    GraphActivationError,
    _before_graph_activate,
    before_graph_activate,
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
async def test_an_infrastructure_failure_keeps_its_text_in_the_log():
    """The message reaches External API callers; a Redis or database client's
    text can name a host, a port or a pool, and isn't about the credential."""
    node = _make_node()
    graph = MagicMock(nodes=[node])

    async def failing_getter(_creds_id):
        raise ConnectionError("Error connecting to redis-master.internal:6379")

    with patch(
        "backend.integrations.webhooks.graph_lifecycle_hooks.credentials_manager"
    ) as mgr:
        mgr.cached_getter.return_value = failing_getter
        with pytest.raises(GraphActivationError) as excinfo:
            await _before_graph_activate(graph, "user-1")

    msg = str(excinfo.value)
    assert "redis" not in msg.lower()
    assert "My GitHub key" in msg
    assert "could not be loaded just now" in msg
    assert "no longer exists" not in msg


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

    # Removed, not set to `{}`: an id-less object fails the field's nested
    # credentials schema at queue time and the node is silently skipped.
    assert "credentials" not in node.input_default


@pytest.mark.asyncio
async def test_an_infrastructure_failure_keeps_an_optional_credential():
    """A transient DB/Redis failure says nothing about the credential, so an
    optional reference is kept and activation fails with a retry message."""
    node = _make_node(required=False, optional_marker=True)
    graph = MagicMock(nodes=[node])

    async def failing_getter(_creds_id):
        raise ConnectionError("Error connecting to redis-master.internal:6379")

    with patch(
        "backend.integrations.webhooks.graph_lifecycle_hooks.credentials_manager"
    ) as mgr:
        mgr.cached_getter.return_value = failing_getter
        with pytest.raises(GraphActivationError) as excinfo:
            await _before_graph_activate(graph, "user-1")

    assert "could not be loaded just now" in str(excinfo.value)
    assert node.input_default["credentials"]["id"] == "cred-1"


@pytest.mark.asyncio
async def test_before_graph_activate_removes_deleted_optional_credentials():
    """A deleted credential on a field with a schema default is removed, so
    the node runs as if none had been picked."""
    node = _make_node(required=False)
    graph = MagicMock(nodes=[node])

    with patch(
        "backend.integrations.webhooks.graph_lifecycle_hooks.credentials_manager"
    ) as mgr:
        mgr.cached_getter.return_value = AsyncMock(return_value=None)
        await _before_graph_activate(graph, "user-1")

    assert "credentials" not in node.input_default


DELETED_CODEX = {
    "id": "deleted-cred",
    "provider": "codex",
    "type": "oauth2",
    "title": "Personal ChatGPT",
}


def _legacy_autopilot_node(**input_default: Any):
    node = MagicMock()
    node.id = "node-1"
    node.block_id = AutoPilotBlock().id
    node.credentials_optional = False
    node.input_default = {"prompt": "do the thing", **input_default}
    node.block.input_schema = AutoPilotBlock.Input
    node.block.name = "AutoPilotBlock"
    return node


async def _activate_with_missing_credential(node) -> None:
    graph = MagicMock(nodes=[node])
    with patch(
        "backend.integrations.webhooks.graph_lifecycle_hooks.credentials_manager"
    ) as mgr:
        mgr.cached_getter.return_value = AsyncMock(return_value=None)
        await _before_graph_activate(graph, "user-1")


async def _run_autopilot_after_save(node):
    """Drive AutoPilotBlock.run on the persisted defaults so we can see
    which account would be billed."""
    block = AutoPilotBlock()
    create_session = AsyncMock(return_value="session-1")
    execute_copilot = AsyncMock(
        return_value=(
            "done",
            [],
            "[]",
            "session-1",
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
    )
    input_data = AutoPilotBlock.Input.model_validate(node.input_default)
    context = ExecutionContext(
        user_id="user-1",
        graph_id="graph-1",
        graph_exec_id="graph-exec-1",
        node_id="node-1",
        node_exec_id="node-exec-1",
    )
    with (
        patch.object(block, "create_session", create_session),
        patch.object(block, "execute_copilot", execute_copilot),
    ):
        outputs = {
            name: value
            async for name, value in block.run(input_data, execution_context=context)
        }
    return outputs, create_session


@pytest.mark.asyncio
async def test_legacy_autopilot_deleted_credential_does_not_switch_to_platform():
    """A submitted legacy AutoPilot node encodes ChatGPT billing as the
    presence of `codex_credentials` and has no `transport`. Clearing a
    deleted reference must not make run() bill platform credits (#16193)."""
    node = _legacy_autopilot_node(codex_credentials=DELETED_CODEX)

    await _activate_with_missing_credential(node)

    assert "codex_credentials" not in node.input_default
    assert (
        node.input_default.get("transport") == AutoPilotTransport.CODEX_APP_SERVER.value
    )

    outputs, create_session = await _run_autopilot_after_save(node)
    assert "error" in outputs
    assert "ChatGPT" in outputs["error"]
    create_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_platform_autopilot_deleted_credential_stays_on_platform():
    """An explicit platform choice is credential-free; a leftover deleted
    Codex reference must still be dropped without rewriting transport."""
    node = _legacy_autopilot_node(
        transport=AutoPilotTransport.PLATFORM.value,
        codex_credentials=DELETED_CODEX,
    )

    await _activate_with_missing_credential(node)

    assert "codex_credentials" not in node.input_default
    assert node.input_default["transport"] == AutoPilotTransport.PLATFORM.value

    outputs, create_session = await _run_autopilot_after_save(node)
    assert outputs["response"] == "done"
    assert create_session.await_args.kwargs["llm_auth_provider"] == "platform"
    assert create_session.await_args.kwargs["llm_credential_id"] is None


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


def _graph_of(*nodes: tuple[str, dict[str, Any]]):
    graph = Graph(
        id="graph-1",
        name="Test",
        description="Test",
        nodes=[
            Node(id=f"node-{i}", block_id=block_id, input_default=input_default)
            for i, (block_id, input_default) in enumerate(nodes)
        ],
    )
    return make_graph_model(graph, user_id="user-1")


def _sheets_node(credentials_id: str) -> tuple[str, dict[str, Any]]:
    spreadsheet = {
        "_credentials_id": credentials_id,
        "id": "sheet-1",
        "name": "test",
        "mimeType": "application/vnd.google-apps.spreadsheet",
    }
    return GoogleSheetsReadBlock().id, {"spreadsheet": spreadsheet, "range": "A1"}


@pytest.mark.asyncio
async def test_before_graph_activate_keeps_only_files_picked_with_own_credentials():
    """[SECRT-1772] A save keeps the file the user picked with their own Google
    account. A file embedding someone else's credentials, as in an agent
    imported from another user's export, is cleared so they pick it again."""
    graph = _graph_of(_sheets_node("own-cred"), _sheets_node("someone-elses-cred"))

    with patch(
        "backend.integrations.webhooks.graph_lifecycle_hooks.credentials_manager"
    ) as mgr:
        mgr.store.get_all_creds = AsyncMock(return_value=[MagicMock(id="own-cred")])
        mgr.cached_getter.return_value = AsyncMock(return_value=None)
        await before_graph_activate(graph, "user-1")

    own, foreign = graph.nodes
    assert own.input_default["spreadsheet"]["_credentials_id"] == "own-cred"
    assert own.input_default["range"] == "A1"
    assert foreign.input_default["spreadsheet"] is None
    mgr.store.get_all_creds.assert_awaited_once_with("user-1")


@pytest.mark.asyncio
async def test_before_graph_activate_skips_the_lookup_without_picked_files():
    """Saving a graph without any picked file must not load the user's
    credentials just to check ownership, and must leave inputs that aren't
    pickers alone even when they hold a `_credentials_id` key."""
    data = {"_credentials_id": "someone-elses-cred"}
    graph = _graph_of((StoreValueBlock().id, {"data": data}))

    with patch(
        "backend.integrations.webhooks.graph_lifecycle_hooks.credentials_manager"
    ) as mgr:
        mgr.store.get_all_creds = AsyncMock(return_value=[])
        mgr.cached_getter.return_value = AsyncMock(return_value=None)
        await before_graph_activate(graph, "user-1")

    mgr.store.get_all_creds.assert_not_awaited()
    assert graph.nodes[0].input_default == {"data": data}


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
