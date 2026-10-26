"""Tests for the shared webhook-preset helpers (setup/update/delete) used by both
the /presets routes and the copilot preset tools."""

import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from backend.api.features.experts import experts_db
from backend.api.features.library.model import node_input_mask_key
from backend.api.features.library.triggers import (
    delete_preset_with_webhook_cleanup,
    setup_triggered_preset,
    update_triggered_preset,
)
from backend.blocks import _base as block_base
from backend.blocks import get_block, get_webhook_block_ids
from backend.blocks.allquiet._webhook import AllQuietWebhooksManager
from backend.blocks.generic_webhook._webhook import GenericWebhooksManager
from backend.blocks.github.triggers import GithubPullRequestTriggerBlock
from backend.blocks.io import AgentNumberInputBlock
from backend.blocks.mcp.block import MCPToolBlock
from backend.data.graph import Graph, GraphModel, Node, make_graph_model
from backend.data.integrations import WebhookWithRelations
from backend.executor import utils as executor_utils
from backend.util.exceptions import InvalidInputError, MissingConfigError, NotFoundError

_USER = "test-user-triggers"
_PATH = "backend.api.features.library.triggers"
_DELIVERY_VALIDATION_CASES = [
    pytest.param(True, False, id="valid-without-credentials"),
    pytest.param("yes", True, id="event-filter-not-boolean"),
]


@pytest.fixture(autouse=True)
def _mock_validate_execution_input():
    """setup_triggered_preset validates inputs via the execution-input builder,
    which hits the DB. Default it to a no-op; the validation-failure test
    overrides it to raise."""
    with patch(
        f"{_PATH}.validate_and_construct_node_execution_input", new=AsyncMock()
    ) as m:
        yield m


def _graph():
    node = MagicMock()
    node.id = "trigger-node"
    graph = MagicMock()
    graph.id = "graph-1"
    graph.version = 1
    graph.webhook_input_node = node
    graph.organization_id = "shared-org"
    graph.team_id = "shared-team"
    return graph


def _patches(*, graph, webhook=..., feedback=None, preset=None, graph_expert=None):
    """Patch triggers.py's collaborators. ``webhook=...`` defaults to a stub
    webhook; pass ``webhook=None`` + ``feedback`` to exercise the rejection.
    ``graph_expert`` is what ``resolve_expert_for_graph`` would return — the
    setup path must never call it (asserted via the returned mock)."""
    new_webhook = MagicMock(id="wh-1") if webhook is ... else webhook
    return (
        patch(f"{_PATH}.get_graph", new=AsyncMock(return_value=graph)),
        patch(f"{_PATH}.make_node_credentials_input_map", return_value={}),
        patch(
            f"{_PATH}.setup_webhook_for_block",
            new=AsyncMock(return_value=(new_webhook, feedback)),
        ),
        patch(f"{_PATH}.db.create_preset", new=AsyncMock(return_value=preset)),
        patch(
            f"{_PATH}.experts_db.resolve_expert_for_graph",
            new=AsyncMock(return_value=graph_expert),
        ),
    )


@pytest.fixture(autouse=True)
def _user_belongs_to_the_graphs_org():
    """A trigger takes its graph's org/team when the user belongs to it.
    Tests of someone else's graph patch this to give the user's own."""
    with patch(
        f"{_PATH}.resolve_member_tenancy",
        new=AsyncMock(side_effect=lambda user_id, org_id, team_id: (org_id, team_id)),
    ) as membership:
        yield membership


def _not_a_member():
    return patch(
        f"{_PATH}.resolve_member_tenancy",
        new=AsyncMock(return_value=("own-org", "own-team")),
    )


async def _setup(*, expert_id: str | None = None):
    return await setup_triggered_preset(
        user_id=_USER,
        graph_id="graph-1",
        graph_version=1,
        name="My Trigger",
        description="",
        trigger_config={"repo": "owner/repo"},
        agent_credentials={},
        expert_id=expert_id,
    )


@pytest.mark.asyncio
async def test_creates_preset_on_success():
    preset = MagicMock(id="preset-1")
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(
        graph=_graph(), preset=preset
    )
    with p_graph, p_creds, p_webhook, p_expert, p_create as create_mock:
        result = await _setup()
    assert result is preset
    create_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_wraps_trigger_config_alongside_constant_inputs():
    """The created preset stores regular inputs plus the trigger config nested
    under a per-node ``_node_input_mask_{node_id}`` key (node id ``trigger-node``
    -> ``trigger``)."""
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(graph=_graph())
    with p_graph, p_creds, p_webhook, p_expert, p_create as create_mock:
        await setup_triggered_preset(
            user_id=_USER,
            graph_id="graph-1",
            graph_version=1,
            name="My Trigger",
            description="",
            trigger_config={"repo": "owner/repo"},
            agent_credentials={},
            constant_inputs={"some_input": "value"},
        )
    created_inputs = create_mock.await_args.kwargs["preset"].inputs
    assert created_inputs == {
        "some_input": "value",
        "_node_input_mask_trigger": {"repo": "owner/repo"},
    }


@pytest.mark.asyncio
async def test_validation_mask_includes_payload_placeholder(
    _mock_validate_execution_input,
):
    """The validation call passes a placeholder `payload` in the trigger mask so
    the webhook trigger node passes execution-input construction (which requires
    a payload for WEBHOOK/WEBHOOK_MANUAL starting nodes)."""
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(graph=_graph())
    with p_graph, p_creds, p_webhook, p_expert, p_create:
        await _setup()
    masks = _mock_validate_execution_input.await_args.kwargs["nodes_input_masks"]
    assert "payload" in masks["trigger-node"]


@pytest.mark.asyncio
async def test_expert_trigger_is_created_in_personal_scope():
    preset = MagicMock(id="preset-1")
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(
        graph=_graph(), preset=preset
    )
    with (
        p_graph,
        p_creds,
        p_expert,
        p_create as create_mock,
        p_webhook as webhook_mock,
        patch(
            f"{_PATH}.experts_db.resolve_private_expert_tenancy",
            new=AsyncMock(return_value=("personal-org", "personal-team")),
        ),
    ):
        result = await _setup(expert_id="expert-1")

    assert result is preset
    assert webhook_mock.await_args.kwargs["organization_id"] == "personal-org"
    assert webhook_mock.await_args.kwargs["team_id"] == "personal-team"
    assert create_mock.await_args.kwargs["expert_id"] == "expert-1"


@pytest.mark.asyncio
async def test_expert_trigger_maps_inaccessible_expert_to_not_found():
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(graph=_graph())
    with (
        p_graph,
        p_creds,
        p_expert,
        p_webhook as webhook_mock,
        p_create as create_mock,
        patch(
            f"{_PATH}.experts_db.resolve_private_expert_tenancy",
            new=AsyncMock(side_effect=experts_db.ExpertNotFoundError("expert-1")),
        ),
    ):
        with pytest.raises(NotFoundError, match="Expert #expert-1 not found"):
            await _setup(expert_id="expert-1")

    webhook_mock.assert_not_awaited()
    create_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_expert_trigger_propagates_transient_tenancy_failure():
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(graph=_graph())
    with (
        p_graph,
        p_creds,
        p_expert,
        p_webhook as webhook_mock,
        p_create as create_mock,
        patch(
            f"{_PATH}.experts_db.resolve_private_expert_tenancy",
            new=AsyncMock(side_effect=RuntimeError("database unavailable")),
        ),
    ):
        with pytest.raises(RuntimeError, match="database unavailable"):
            await _setup(expert_id="expert-1")

    webhook_mock.assert_not_awaited()
    create_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_expert_trigger_maps_missing_personal_tenancy_to_unavailable():
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(graph=_graph())
    with (
        p_graph,
        p_creds,
        p_expert,
        p_webhook as webhook_mock,
        p_create as create_mock,
        patch(
            f"{_PATH}.experts_db.resolve_private_expert_tenancy",
            new=AsyncMock(
                side_effect=experts_db.ExpertPrivateTenancyNotFoundError("expert-1")
            ),
        ),
    ):
        with pytest.raises(MissingConfigError, match="still being set up"):
            await _setup(expert_id="expert-1")

    webhook_mock.assert_not_awaited()
    create_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_setup_prefers_session_expert_over_graph_match():
    """A trigger created inside an expert chat requests that expert directly,
    short-circuiting the graph-match fallback."""
    preset = MagicMock(id="preset-1")
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(
        graph=_graph(), preset=preset, graph_expert="expert-graph"
    )
    with (
        p_graph,
        p_creds,
        p_webhook,
        p_expert as expert_mock,
        p_create as create_mock,
        patch(
            f"{_PATH}.experts_db.resolve_private_expert_tenancy",
            new=AsyncMock(return_value=("personal-org", "personal-team")),
        ),
    ):
        await _setup(expert_id="expert-session")
    assert create_mock.call_args.kwargs["expert_id"] == "expert-session"
    expert_mock.assert_not_awaited()  # short-circuited by the session expert


@pytest.mark.asyncio
async def test_setup_supplied_session_expert_never_graph_reattributes():
    """A supplied session expert goes straight to the private-tenancy guard.
    Even when it has become invalid, this layer must not silently replace
    Expert A with a graph match for Expert B — it fails closed instead."""
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(
        graph=_graph(), graph_expert="expert-graph"
    )
    with (
        p_graph,
        p_creds,
        p_webhook,
        p_expert as expert_mock,
        p_create as create_mock,
        patch(
            f"{_PATH}.experts_db.resolve_private_expert_tenancy",
            new=AsyncMock(
                side_effect=experts_db.ExpertNotFoundError("expert-archived")
            ),
        ),
    ):
        with pytest.raises(NotFoundError, match="Expert #expert-archived not found"):
            await _setup(expert_id="expert-archived")
    create_mock.assert_not_awaited()
    expert_mock.assert_not_awaited()  # no graph-match re-attribution


@pytest.mark.asyncio
async def test_setup_without_expert_creates_unattributed_preset():
    """``expert_id=None`` means exactly that: an Otto copilot session
    must get an unattributed preset it can manage from its own scope, so this
    layer never infers an expert from the graph. Callers that want graph-match
    attribution (the HTTP route) resolve it themselves before calling in."""
    preset = MagicMock(id="preset-1")
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(
        graph=_graph(), preset=preset, graph_expert="expert-graph"
    )
    with (
        p_graph,
        p_creds,
        p_webhook,
        p_expert as expert_mock,
        p_create as create_mock,
    ):
        await _setup()
    assert create_mock.call_args.kwargs["expert_id"] is None
    expert_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_not_found_raises():
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(graph=None)
    with p_graph, p_creds, p_webhook, p_create, p_expert:
        with pytest.raises(NotFoundError):
            await _setup()


@pytest.mark.asyncio
async def test_no_webhook_node_raises():
    graph = _graph()
    graph.webhook_input_node = None
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(graph=graph)
    with p_graph, p_creds, p_webhook, p_create, p_expert:
        with pytest.raises(InvalidInputError):
            await _setup()


@pytest.mark.asyncio
async def test_webhook_setup_rejected_raises():
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(
        graph=_graph(), webhook=None, feedback="no enabled events"
    )
    with p_graph, p_creds, p_webhook, p_expert, p_create as create_mock:
        with pytest.raises(InvalidInputError, match="no enabled events"):
            await _setup()
    create_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_inputs_rejected_before_webhook_setup():
    """Inputs the execution-input validator rejects fail the setup before any
    webhook is registered or preset created."""
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(graph=_graph())
    with (
        p_graph,
        p_creds,
        p_expert,
        p_webhook as webhook_mock,
        p_create as create_mock,
        patch(
            f"{_PATH}.validate_and_construct_node_execution_input",
            new=AsyncMock(side_effect=ValueError("count is not a number")),
        ),
    ):
        with pytest.raises(InvalidInputError, match="Invalid preset inputs"):
            await setup_triggered_preset(
                user_id=_USER,
                graph_id="graph-1",
                graph_version=1,
                name="My Trigger",
                description="",
                trigger_config={"repo": "owner/repo"},
                agent_credentials={},
                constant_inputs={"count": "not-a-number"},
            )
    webhook_mock.assert_not_awaited()
    create_mock.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("opened, refused", _DELIVERY_VALIDATION_CASES)
async def test_setup_refuses_what_delivery_refuses(
    trigger_blocks_enabled, opened, refused
):
    """Setup runs the executor's validation, so a trigger config delivery would
    refuse is refused before the webhook exists; only credentials may be unset."""
    graph = _trigger_graph(GithubPullRequestTriggerBlock().id)
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(graph=graph)
    with (
        p_graph,
        p_creds,
        p_expert,
        p_create,
        p_webhook as webhook_mock,
        _real_execution_input_validator(graph),
        _refusal(refused),
    ):
        await setup_triggered_preset(
            user_id=_USER,
            graph_id=graph.id,
            graph_version=graph.version,
            name="PR agent",
            description="",
            trigger_config={"repo": "owner/repo", "events": {"opened": opened}},
            agent_credentials={},
            constant_inputs={"count": 3},
        )
    assert webhook_mock.await_count == (0 if refused else 1)


@pytest.mark.asyncio
async def test_setup_accepts_every_trigger_block_before_its_first_event(
    trigger_blocks_enabled,
):
    """No event exists at setup, so the placeholder payload must pass for every
    trigger block, including one whose payload is a typed model (Compass)."""
    checked, refused = [], {}
    for block_id in get_webhook_block_ids():
        if (block := get_block(block_id)).disabled:
            continue
        graph = _trigger_graph(block_id)
        p_graph, p_creds, p_webhook, p_create, p_expert = _patches(graph=graph)
        with (
            p_graph,
            p_creds,
            p_webhook,
            p_create,
            p_expert,
            _real_execution_input_validator(graph),
        ):
            try:
                await setup_triggered_preset(
                    user_id=_USER,
                    graph_id=graph.id,
                    graph_version=graph.version,
                    name=block.name,
                    description="",
                    trigger_config=_example_trigger_config(block),
                    agent_credentials={},
                    constant_inputs={"count": 3},
                )
            except InvalidInputError as e:
                refused[block.name] = str(e)
        checked.append(block.name)
    assert "CompassAITriggerBlock" in checked
    assert refused == {}


@pytest.mark.asyncio
async def test_setup_keeps_a_blocks_own_input_check(trigger_blocks_enabled):
    """An MCP tool beside the trigger, without credentials yet: delivery checks
    it with the block's own get_mismatch_error, so setup must as well."""
    mcp_node = Node(
        id="mcp",
        block_id=MCPToolBlock().id,
        input_default={
            "server_url": "https://mcp.example.com/mcp",
            "selected_tool": "search",
            "tool_input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            "tool_arguments": {"query": "triggers"},
        },
    )
    graph = _trigger_graph(GithubPullRequestTriggerBlock().id, mcp_node)
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(graph=graph)
    with (
        p_graph,
        p_creds,
        p_expert,
        p_create,
        p_webhook as webhook_mock,
        _real_execution_input_validator(graph),
    ):
        await setup_triggered_preset(
            user_id=_USER,
            graph_id=graph.id,
            graph_version=graph.version,
            name="PR agent",
            description="",
            trigger_config={"repo": "owner/repo", "events": {"opened": True}},
            agent_credentials={},
            constant_inputs={"count": 3},
        )
    webhook_mock.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "manager, secret_field",
    [
        (GenericWebhooksManager, "secret_token"),
        (AllQuietWebhooksManager, "signing_secret"),
    ],
)
@pytest.mark.parametrize(
    "constant_inputs",
    [
        {"topic": "news"},
        {"_node_input_mask_a": "news"},
        {"_node_input_mask_a": {"topic": "news"}},
    ],
    ids=["plain-input", "prefixed-string-input", "prefixed-object-input"],
)
async def test_stored_trigger_secret_still_guards_ingress(
    manager, secret_field, constant_inputs
):
    """A secret set at setup is found where setup stored it, even beside a graph
    input that uses the mask prefix, so the webhook refuses an unsigned delivery."""
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(graph=_graph())
    with p_graph, p_creds, p_webhook, p_expert, p_create as create_mock:
        await setup_triggered_preset(
            user_id=_USER,
            graph_id="graph-1",
            graph_version=1,
            name="Signed Trigger",
            description="",
            trigger_config={secret_field: "t0ken"},
            agent_credentials={},
            constant_inputs=constant_inputs,
        )
    stored = create_mock.await_args.kwargs["preset"].inputs
    assert next(iter(stored)) in constant_inputs
    webhook = WebhookWithRelations.model_construct(
        id="wh-1",
        triggered_nodes=[],
        triggered_presets=[SimpleNamespace(inputs=stored)],
    )
    unsigned = Request({"type": "http", "method": "POST", "headers": []})
    with pytest.raises(HTTPException) as exc:
        await manager.verify_signature(webhook, unsigned)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_a_trigger_on_someone_elses_workflow_stays_in_the_users_org():
    """The workflow can be in an org the user isn't in. A trigger they set up
    on it, and the runs it fires, stay in their own."""
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(
        graph=_graph(), preset=MagicMock(id="preset-1")
    )
    with (
        _not_a_member() as membership,
        p_graph,
        p_creds,
        p_create,
        p_expert,
        p_webhook as webhook_mock,
    ):
        await _setup()
    membership.assert_awaited_once_with(_USER, "shared-org", "shared-team")
    kwargs = webhook_mock.await_args.kwargs
    assert (kwargs["organization_id"], kwargs["team_id"]) == ("own-org", "own-team")


# ---- update_triggered_preset ----


def _preset(
    *,
    webhook_id=None,
    expert_id=None,
    organization_id="shared-org",
    team_id="shared-team",
):
    preset = MagicMock()
    preset.id = "preset-1"
    preset.graph_id = "graph-1"
    preset.graph_version = 1
    preset.webhook_id = webhook_id
    preset.expert_id = expert_id
    preset.organization_id = organization_id
    preset.team_id = team_id
    preset.inputs = {}
    return preset


@contextlib.contextmanager
def _update_patches(*, current, graph=..., webhook=..., feedback=None):
    """Patch update_triggered_preset's collaborators; yields the call mocks."""
    new_webhook = MagicMock(id="wh-new") if webhook is ... else webhook
    graph_val = _graph() if graph is ... else graph
    patchers = {
        "get_preset": patch(
            f"{_PATH}.db.get_preset", new=AsyncMock(return_value=current)
        ),
        "get_graph": patch(f"{_PATH}.get_graph", new=AsyncMock(return_value=graph_val)),
        "creds_map": patch(f"{_PATH}.make_node_credentials_input_map", return_value={}),
        "setup": patch(
            f"{_PATH}.setup_webhook_for_block",
            new=AsyncMock(return_value=(new_webhook, feedback)),
        ),
        "update": patch(
            f"{_PATH}.db.update_preset",
            new=AsyncMock(return_value=MagicMock(id="preset-1")),
        ),
        "set_webhook": patch(
            f"{_PATH}.db.set_preset_webhook",
            new=AsyncMock(return_value=MagicMock(id="preset-1")),
        ),
        "prune": patch(f"{_PATH}._prune_dangling_webhook", new=AsyncMock()),
    }
    with contextlib.ExitStack() as stack:
        yield {k: stack.enter_context(p) for k, p in patchers.items()}


@pytest.mark.asyncio
async def test_update_rename_only_skips_webhook():
    with _update_patches(current=_preset(webhook_id="wh-old")) as m:
        await update_triggered_preset(
            user_id=_USER, preset_id="preset-1", name="New name"
        )
    m["update"].assert_awaited_once()
    m["setup"].assert_not_awaited()
    m["set_webhook"].assert_not_awaited()
    m["prune"].assert_not_awaited()


@pytest.mark.asyncio
async def test_update_reconfigure_reregisters_and_prunes_old():
    with _update_patches(current=_preset(webhook_id="wh-old")) as m:
        await update_triggered_preset(
            user_id=_USER,
            preset_id="preset-1",
            inputs={"_node_input_mask_trigger": {"repo": "owner/repo"}},
            credentials={},
        )
    m["setup"].assert_awaited_once()
    m["set_webhook"].assert_awaited_once()
    m["prune"].assert_awaited_once_with(_USER, "wh-old")


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["update", "set_webhook"])
@pytest.mark.parametrize("registered, pruned", [("wh-new", True), ("wh-old", False)])
async def test_a_failed_update_prunes_only_the_webhook_it_registered(
    failing, registered, pruned
):
    """The new webhook is attached last, so a failure before that would leave
    it registered with the provider and attached to nothing."""
    current = _preset(webhook_id="wh-old")
    with _update_patches(current=current, webhook=MagicMock(id=registered)) as m:
        m[failing].side_effect = RuntimeError("database down")
        with pytest.raises(RuntimeError):
            await update_triggered_preset(
                user_id=_USER,
                preset_id="preset-1",
                inputs={"repo": "owner/repo"},
                credentials={},
            )
    assert m["prune"].await_args_list == ([call(_USER, registered)] if pruned else [])


@pytest.mark.asyncio
async def test_reconfiguring_a_trigger_on_someone_elses_workflow_registers_it_in_the_users_org():
    """The webhook moves to the user's own org and still gets only the trigger
    config, with the regular inputs stored beside it."""
    inputs = {"topic": "news", "_node_input_mask_trigger": {"repo": "owner/repo"}}
    with _not_a_member(), _update_patches(current=_preset(webhook_id="wh-old")) as m:
        await update_triggered_preset(
            user_id=_USER, preset_id="preset-1", inputs=inputs, credentials={}
        )
    kwargs = m["setup"].await_args.kwargs
    assert (kwargs["organization_id"], kwargs["team_id"]) == ("own-org", "own-team")
    assert kwargs["trigger_config"] == {"repo": "owner/repo"}
    assert m["update"].await_args.kwargs["inputs"] == inputs


@pytest.mark.asyncio
async def test_update_expert_trigger_keeps_personal_scope():
    current = _preset(
        webhook_id="wh-old",
        expert_id="expert-1",
        organization_id="personal-org",
        team_id="personal-team",
    )
    with (
        _update_patches(current=current) as m,
        patch(
            f"{_PATH}.experts_db.resolve_private_expert_tenancy",
            new=AsyncMock(return_value=("personal-org", "personal-team")),
        ),
    ):
        await update_triggered_preset(
            user_id=_USER,
            preset_id="preset-1",
            inputs={"_node_input_mask_trigger": {"repo": "owner/repo"}},
            credentials={},
        )

    assert m["setup"].await_args.kwargs["organization_id"] == "personal-org"
    assert m["setup"].await_args.kwargs["team_id"] == "personal-team"


@pytest.mark.asyncio
async def test_update_legacy_expert_trigger_rehomes_through_new_webhook():
    current = _preset(webhook_id="wh-old", expert_id="expert-1")
    with (
        _update_patches(current=current) as m,
        patch(
            f"{_PATH}.experts_db.resolve_private_expert_tenancy",
            new=AsyncMock(return_value=("personal-org", "personal-team")),
        ),
    ):
        await update_triggered_preset(
            user_id=_USER,
            preset_id="preset-1",
            inputs={"_node_input_mask_trigger": {"repo": "owner/repo"}},
            credentials={},
        )

    assert m["setup"].await_args.kwargs["organization_id"] == "personal-org"
    assert m["setup"].await_args.kwargs["team_id"] == "personal-team"
    m["set_webhook"].assert_awaited_once_with(_USER, "preset-1", "wh-new")


@pytest.mark.asyncio
async def test_update_expert_trigger_maps_missing_personal_tenancy_to_unavailable():
    current = _preset(webhook_id="wh-old", expert_id="expert-1")
    with (
        _update_patches(current=current) as m,
        patch(
            f"{_PATH}.experts_db.resolve_private_expert_tenancy",
            new=AsyncMock(
                side_effect=experts_db.ExpertPrivateTenancyNotFoundError("expert-1")
            ),
        ),
    ):
        with pytest.raises(MissingConfigError, match="still being set up"):
            await update_triggered_preset(
                user_id=_USER,
                preset_id="preset-1",
                inputs={"_node_input_mask_trigger": {"repo": "owner/repo"}},
                credentials={},
            )

    m["setup"].assert_not_awaited()
    m["update"].assert_not_awaited()


@pytest.mark.asyncio
async def test_update_reconfigure_webhook_rejected_raises():
    with _update_patches(current=_preset(), webhook=None, feedback="no events") as m:
        with pytest.raises(InvalidInputError, match="no events"):
            await update_triggered_preset(
                user_id=_USER,
                preset_id="preset-1",
                inputs={"_node_input_mask_trigger": {"repo": "x"}},
                credentials={},
            )
    m["update"].assert_not_awaited()


@pytest.mark.asyncio
async def test_update_reconfigure_validates_inputs_before_registering(
    _mock_validate_execution_input,
):
    """Regular graph inputs are editable on a triggered preset, so the update
    path must reject an invalid set before the webhook is re-registered —
    otherwise every later delivery dies inside the executor's catch-all."""
    _mock_validate_execution_input.side_effect = ValueError("field required: topic")
    with _update_patches(current=_preset(webhook_id="wh-old")) as m:
        with pytest.raises(InvalidInputError, match="Invalid preset inputs"):
            await update_triggered_preset(
                user_id=_USER,
                preset_id="preset-1",
                inputs={"_node_input_mask_trigger": {"repo": "x"}, "topic": None},
                credentials={},
            )
    m["setup"].assert_not_awaited()
    m["update"].assert_not_awaited()


@pytest.mark.asyncio
async def test_update_reconfigure_validates_mask_apart_from_graph_inputs(
    _mock_validate_execution_input,
):
    """The trigger config goes to the node mask, everything else to graph
    inputs — the same split the executor makes at delivery time."""
    with _update_patches(current=_preset(webhook_id="wh-old")):
        await update_triggered_preset(
            user_id=_USER,
            preset_id="preset-1",
            inputs={
                "_node_input_mask_trigger": {"repo": "owner/repo"},
                "topic": "weather",
            },
            credentials={},
        )
    kwargs = _mock_validate_execution_input.await_args.kwargs
    assert kwargs["graph_inputs"] == {"topic": "weather"}
    assert kwargs["nodes_input_masks"] == {
        "trigger-node": {"repo": "owner/repo", "payload": {}}
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("opened, refused", _DELIVERY_VALIDATION_CASES)
async def test_reconfigure_refuses_what_delivery_refuses(
    trigger_blocks_enabled, opened, refused
):
    graph = _trigger_graph(GithubPullRequestTriggerBlock().id)
    trigger_config = {"repo": "owner/repo", "events": {"opened": opened}}
    with (
        _update_patches(current=_preset(webhook_id="wh-old"), graph=graph) as m,
        _real_execution_input_validator(graph),
        _refusal(refused),
    ):
        await update_triggered_preset(
            user_id=_USER,
            preset_id="preset-1",
            inputs={"count": 3, node_input_mask_key("trigger"): trigger_config},
            credentials={},
        )
    assert m["setup"].await_count == (0 if refused else 1)


@pytest.mark.asyncio
async def test_update_reconfigure_stores_new_credentials_in_mask():
    """The stored mask wins over preset.credentials at execution, so replaced
    credentials the caller echoed back in the mask must not survive the update."""
    old_creds = {"id": "cred-old", "provider": "github", "type": "api_key"}
    new_creds = {"id": "cred-new", "provider": "github", "type": "api_key"}
    with _update_patches(current=_preset(webhook_id="wh-old")) as m:
        m["creds_map"].return_value = {"trigger-node": {"credentials": new_creds}}
        await update_triggered_preset(
            user_id=_USER,
            preset_id="preset-1",
            inputs={
                "_node_input_mask_trigger": {
                    "repo": "owner/repo",
                    "credentials": old_creds,
                },
                "topic": "weather",
            },
            credentials={"credentials": MagicMock()},
        )
    assert m["update"].await_args.kwargs["inputs"] == {
        "_node_input_mask_trigger": {"repo": "owner/repo", "credentials": new_creds},
        "topic": "weather",
    }


@pytest.mark.asyncio
async def test_update_run_template_on_trigger_graph_skips_webhook():
    """A graph can hold input nodes beside its trigger node, so a run-template
    preset on it carries no mask and no webhook; its edit is a plain update."""
    with _update_patches(current=_preset()) as m:
        await update_triggered_preset(
            user_id=_USER,
            preset_id="preset-1",
            inputs={"topic": "weather"},
            credentials={},
        )
    m["setup"].assert_not_awaited()
    m["set_webhook"].assert_not_awaited()
    assert m["update"].await_args.kwargs["inputs"] == {"topic": "weather"}


@pytest.mark.asyncio
@pytest.mark.parametrize("inputs", [{"some_input": "value"}, {}])
async def test_update_reconfigure_missing_input_mask_raises(inputs):
    """Reconfiguring a nested triggered preset without the
    ``_node_input_mask_{node_id}`` key is rejected before any webhook work, an
    update clearing every input included: that would delete the trigger config."""
    current = _preset(webhook_id="wh-old")
    current.inputs = {"_node_input_mask_trigger": {"repo": "owner/repo"}}
    with _update_patches(current=current) as m:
        with pytest.raises(InvalidInputError, match="Missing trigger configuration"):
            await update_triggered_preset(
                user_id=_USER,
                preset_id="preset-1",
                inputs=inputs,
                credentials={},
            )
    m["setup"].assert_not_awaited()
    m["update"].assert_not_awaited()


@pytest.mark.asyncio
async def test_clearing_a_triggers_graph_inputs_keeps_its_trigger_config():
    current = _preset(webhook_id="wh-old")
    current.inputs = {
        "topic": "weather",
        "_node_input_mask_trigger": {"repo": "owner/repo"},
    }
    with _update_patches(current=current) as m:
        await update_triggered_preset(
            user_id=_USER,
            preset_id="preset-1",
            inputs={"_node_input_mask_trigger": {"repo": "owner/repo"}},
            credentials={},
        )
    assert m["update"].await_args.kwargs["inputs"] == {
        "_node_input_mask_trigger": {"repo": "owner/repo"}
    }
    m["set_webhook"].assert_awaited_once()


@pytest.mark.asyncio
async def test_update_reconfigure_flat_preset_is_stored_nested():
    """A flat preset the boot backfill has not converted yet is read as delivery
    reads it — all inputs are trigger config — and saved in the nested shape."""
    current = _preset(webhook_id="wh-old")
    current.inputs = {"repo": "owner/old"}
    with _update_patches(current=current) as m:
        await update_triggered_preset(
            user_id=_USER,
            preset_id="preset-1",
            inputs={"repo": "owner/new"},
            credentials={},
        )
    assert m["setup"].await_args.kwargs["trigger_config"] == {"repo": "owner/new"}
    assert m["update"].await_args.kwargs["inputs"] == {
        "_node_input_mask_trigger": {"repo": "owner/new"}
    }
    m["set_webhook"].assert_awaited_once()


@pytest.mark.asyncio
async def test_update_reconfigure_non_dict_trigger_config_raises():
    """A non-dict value under the trigger mask key is rejected with a clean
    InvalidInputError rather than a TypeError from `**trigger_config`."""
    with _update_patches(current=_preset(webhook_id="wh-old")) as m:
        with pytest.raises(InvalidInputError, match="must be an object"):
            await update_triggered_preset(
                user_id=_USER,
                preset_id="preset-1",
                inputs={"_node_input_mask_trigger": "not-a-dict"},
                credentials={},
            )
    m["setup"].assert_not_awaited()
    m["update"].assert_not_awaited()


@pytest.mark.asyncio
async def test_update_preset_not_found_raises():
    with _update_patches(current=None):
        with pytest.raises(NotFoundError):
            await update_triggered_preset(user_id=_USER, preset_id="missing", name="X")


@pytest.mark.asyncio
async def test_update_reconfigure_graph_gone_raises():
    with _update_patches(current=_preset(), graph=None):
        with pytest.raises(NotFoundError):
            await update_triggered_preset(
                user_id=_USER,
                preset_id="preset-1",
                inputs={"repo": "x"},
                credentials={},
            )


# ---- delete_preset_with_webhook_cleanup ----


@contextlib.contextmanager
def _delete_patches(*, preset):
    patchers = {
        "get_preset": patch(
            f"{_PATH}.db.get_preset", new=AsyncMock(return_value=preset)
        ),
        "set_webhook": patch(f"{_PATH}.db.set_preset_webhook", new=AsyncMock()),
        "prune": patch(f"{_PATH}._prune_dangling_webhook", new=AsyncMock()),
        "delete": patch(f"{_PATH}.db.delete_preset", new=AsyncMock()),
    }
    with contextlib.ExitStack() as stack:
        yield {k: stack.enter_context(p) for k, p in patchers.items()}


@pytest.mark.asyncio
async def test_delete_with_webhook_prunes():
    with _delete_patches(preset=_preset(webhook_id="wh-1")) as m:
        await delete_preset_with_webhook_cleanup(user_id=_USER, preset_id="preset-1")
    m["set_webhook"].assert_awaited_once_with(_USER, "preset-1", None)
    m["prune"].assert_awaited_once_with(_USER, "wh-1")
    m["delete"].assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_without_webhook_skips_prune():
    with _delete_patches(preset=_preset(webhook_id=None)) as m:
        await delete_preset_with_webhook_cleanup(user_id=_USER, preset_id="preset-1")
    m["set_webhook"].assert_not_awaited()
    m["prune"].assert_not_awaited()
    m["delete"].assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_not_found_raises():
    with _delete_patches(preset=None):
        with pytest.raises(NotFoundError):
            await delete_preset_with_webhook_cleanup(user_id=_USER, preset_id="missing")


@pytest.mark.asyncio
async def test_prune_dangling_webhook_is_best_effort():
    """Cleanup runs after state is committed, so a prune failure is swallowed
    (logged) rather than raised — the mutation must not fail post-commit."""
    from backend.api.features.library.triggers import _prune_dangling_webhook

    with patch(
        f"{_PATH}.get_webhook",
        new=AsyncMock(side_effect=Exception("provider unreachable")),
    ):
        # Must not raise.
        await _prune_dangling_webhook(_USER, "wh-1")


@pytest.fixture
def trigger_blocks_enabled(monkeypatch):
    # Trigger blocks are disabled without a platform URL, which CI does not set.
    monkeypatch.setattr(
        block_base.app_config, "platform_base_url", "https://example.com"
    )


def _trigger_graph(trigger_block_id: str, *other_nodes: Node) -> GraphModel:
    """A trigger beside a number input, as the executor sees it."""
    return make_graph_model(
        Graph(
            id="graph-1",
            version=1,
            name="PR agent",
            description="",
            nodes=[
                Node(id="trigger", block_id=trigger_block_id),
                Node(
                    id="count",
                    block_id=AgentNumberInputBlock().id,
                    input_default={"name": "count"},
                ),
                *other_nodes,
            ],
            links=[],
        ),
        _USER,
    )


@contextlib.contextmanager
def _real_execution_input_validator(graph: GraphModel):
    """Undo the autouse stub: run the executor's validation with only its DB
    reads stubbed."""
    with (
        patch(
            f"{_PATH}.validate_and_construct_node_execution_input",
            new=executor_utils.validate_and_construct_node_execution_input,
        ),
        patch.object(executor_utils, "prisma", MagicMock(is_connected=lambda: True)),
        patch.object(
            executor_utils.graph_db, "get_graph", new=AsyncMock(return_value=graph)
        ),
        patch.object(
            executor_utils.graph_db,
            "validate_graph_execution_permissions",
            new=AsyncMock(),
        ),
    ):
        yield


def _refusal(refused: bool):
    if not refused:
        return contextlib.nullcontext()
    return pytest.raises(InvalidInputError, match="'yes' is not of type 'boolean'")


def _example_trigger_config(block) -> dict:
    example = (
        block.test_input[0] if isinstance(block.test_input, list) else block.test_input
    )
    credentials = block.input_schema.get_credentials_fields()
    return {
        k: v
        for k, v in (example or {}).items()
        if k != "payload" and k not in credentials
    }
