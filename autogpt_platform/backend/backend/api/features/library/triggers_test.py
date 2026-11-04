"""Tests for the shared webhook-preset helpers (setup/update/delete) used by both
the /presets routes and the copilot preset tools."""

import contextlib
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from backend.api.features.experts import experts_db
from backend.api.features.library import db
from backend.api.features.library.triggers import (
    delete_preset_with_webhook_cleanup,
    setup_triggered_preset,
    update_triggered_preset,
)
from backend.data.includes import MAX_NODE_INPUT_OUTPUT_FETCH
from backend.data.model import CredentialsMetaInput
from backend.util.exceptions import InvalidInputError, MissingConfigError, NotFoundError

_USER = "test-user-triggers"
_PATH = "backend.api.features.library.triggers"


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
def _under_the_preset_cap():
    with patch(f"{_PATH}.db.check_preset_count", new=AsyncMock()) as check:
        yield check


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


async def _setup(*, expert_id: str | None = None, trigger_config: dict | None = None):
    return await setup_triggered_preset(
        user_id=_USER,
        graph_id="graph-1",
        graph_version=1,
        name="My Trigger",
        description="",
        trigger_config=(
            {"repo": "owner/repo"} if trigger_config is None else trigger_config
        ),
        agent_credentials={},
        expert_id=expert_id,
    )


@pytest.mark.asyncio
async def test_a_refused_trigger_config_registers_no_webhook():
    """Nothing is registered with the provider for a value the write refuses.

    The preset write bounds its own text and inputs, so without this check the
    webhook would be registered and then left behind when that write refuses.
    """
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(graph=_graph())
    with p_graph, p_creds, p_expert, p_create, p_webhook as webhook_mock:
        with pytest.raises(InvalidInputError, match="Preset name is too long"):
            await setup_triggered_preset(
                user_id=_USER,
                graph_id="graph-1",
                graph_version=1,
                name="x" * (db.MAX_PRESET_NAME_LENGTH + 1),
                description="",
                trigger_config={"repo": "owner/repo"},
                agent_credentials={},
            )
    webhook_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_trigger_config_over_the_byte_bound_registers_no_webhook():
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(graph=_graph())
    with p_graph, p_creds, p_expert, p_create, p_webhook as webhook_mock:
        with pytest.raises(InvalidInputError, match="inputs hold at most"):
            await setup_triggered_preset(
                user_id=_USER,
                graph_id="graph-1",
                graph_version=1,
                name="My Trigger",
                description="",
                trigger_config={"repo": "x" * (db.MAX_PRESET_INPUT_BYTES + 1)},
                agent_credentials={},
            )
    webhook_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_webhook_is_pruned_when_the_preset_write_fails():
    """Whatever the write fails on, the webhook does not outlive the attempt."""
    p_graph, p_creds, p_webhook, _p_create, p_expert = _patches(graph=_graph())
    with (
        p_graph,
        p_creds,
        p_expert,
        p_webhook,
        patch(
            f"{_PATH}.db.create_preset",
            new=AsyncMock(side_effect=RuntimeError("the write failed")),
        ),
        patch(f"{_PATH}._prune_dangling_webhook", new=AsyncMock()) as prune_mock,
    ):
        with pytest.raises(RuntimeError, match="the write failed"):
            await _setup()
    prune_mock.assert_awaited_once_with(_USER, "wh-1")


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
async def test_an_oversized_trigger_config_registers_no_webhook():
    """Refused before the webhook is registered, so none is left to clean up."""
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(graph=_graph())
    oversized = {f"field_{i}": i for i in range(MAX_NODE_INPUT_OUTPUT_FETCH + 1)}
    with (
        p_graph,
        p_creds,
        p_expert,
        p_webhook as webhook_mock,
        p_create as create_mock,
        pytest.raises(InvalidInputError, match="at most"),
    ):
        await _setup(trigger_config=oversized)
    webhook_mock.assert_not_awaited()
    create_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_trigger_config_filled_to_the_bound_registers_no_webhook():
    """Webhook ingress adds `payload` to the trigger node, so a configuration
    filled right up to the bound would put one name more than the bound on that
    node and its runs could never start. Refused when it is saved instead."""
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(graph=_graph())
    at_the_bound = {f"field_{i}": i for i in range(MAX_NODE_INPUT_OUTPUT_FETCH)}
    with (
        p_graph,
        p_creds,
        p_expert,
        p_webhook as webhook_mock,
        p_create as create_mock,
        pytest.raises(InvalidInputError, match="at most"),
    ):
        await _setup(trigger_config=at_the_bound)
    webhook_mock.assert_not_awaited()
    create_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_trigger_config_at_the_bound_carrying_payload_is_accepted():
    """The reserved slot is `payload`'s own, not an extra one: a configuration
    already naming it is stored at the bound, as the writer allows."""
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(graph=_graph())
    at_the_bound = {f"field_{i}": i for i in range(MAX_NODE_INPUT_OUTPUT_FETCH - 1)}
    at_the_bound["payload"] = None
    with p_graph, p_creds, p_expert, p_webhook, p_create as create_mock:
        await _setup(trigger_config=at_the_bound)
    create_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_an_account_at_the_preset_cap_registers_no_webhook(
    _under_the_preset_cap,
):
    _under_the_preset_cap.side_effect = InvalidInputError("at most 1000 presets")
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(graph=_graph())
    with (
        p_graph,
        p_creds,
        p_expert,
        p_webhook as webhook_mock,
        p_create as create_mock,
        pytest.raises(InvalidInputError, match="at most"),
    ):
        await _setup()
    _under_the_preset_cap.assert_awaited_once_with(_USER)
    webhook_mock.assert_not_awaited()
    create_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_trigger_config_oversized_only_once_every_credential_is_counted_registers_no_webhook():
    """The preflight has to count what create_preset will actually store —
    the trigger config merged with EVERY agent credential, not only the
    trigger node's — or a config oversized that way slips past it and
    registers a webhook that create_preset then refuses."""
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(graph=_graph())
    small_trigger_config = {"repo": "owner/repo"}
    credentials_from_other_nodes = {
        f"cred_{i}": CredentialsMetaInput(
            id=f"cred-{i}", provider="github", type="api_key"
        )
        for i in range(MAX_NODE_INPUT_OUTPUT_FETCH)
    }
    with (
        p_graph,
        p_creds,
        p_expert,
        p_webhook as webhook_mock,
        p_create as create_mock,
        pytest.raises(InvalidInputError, match="at most"),
    ):
        await setup_triggered_preset(
            user_id=_USER,
            graph_id="graph-1",
            graph_version=1,
            name="My Trigger",
            description="",
            trigger_config=small_trigger_config,
            agent_credentials=credentials_from_other_nodes,
        )
    webhook_mock.assert_not_awaited()
    create_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_setup_prunes_the_webhook_when_create_preset_fails():
    """create_preset can still fail after the webhook is registered (e.g. a
    graph access change landing in between); the webhook just registered for
    this setup must not be left dangling."""
    p_graph, p_creds, p_webhook, _p_create, p_expert = _patches(graph=_graph())
    with (
        p_graph,
        p_creds,
        p_expert,
        p_webhook as webhook_mock,
        patch(
            f"{_PATH}.db.create_preset",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ),
        patch(f"{_PATH}._prune_dangling_webhook", new=AsyncMock()) as prune,
        pytest.raises(RuntimeError, match="boom"),
    ):
        await _setup()
    webhook_mock.assert_awaited_once()
    prune.assert_awaited_once_with(_USER, "wh-1")


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
async def test_an_oversized_trigger_update_registers_no_webhook():
    """Refused before the webhook is registered, so none is left to prune."""
    oversized = {f"field_{i}": i for i in range(MAX_NODE_INPUT_OUTPUT_FETCH + 1)}
    with _update_patches(current=_preset(webhook_id="wh-old")) as m:
        with pytest.raises(InvalidInputError, match="at most"):
            await update_triggered_preset(
                user_id=_USER,
                preset_id="preset-1",
                inputs=oversized,
                credentials={},
            )
    m["setup"].assert_not_awaited()
    m["update"].assert_not_awaited()
    m["prune"].assert_not_awaited()


@pytest.mark.asyncio
async def test_an_update_oversized_only_once_every_credential_is_counted_registers_no_webhook():
    """The preflight has to count what update_preset will actually store —
    inputs merged with every credential, not only the trigger node's — or a
    configuration oversized that way slips past it and registers a webhook
    that update_preset then refuses."""
    small_inputs = {"repo": "owner/repo"}
    credentials_from_other_nodes = {
        f"cred_{i}": CredentialsMetaInput(
            id=f"cred-{i}", provider="github", type="api_key"
        )
        for i in range(MAX_NODE_INPUT_OUTPUT_FETCH)
    }
    with _update_patches(current=_preset(webhook_id="wh-old")) as m:
        with pytest.raises(InvalidInputError, match="at most"):
            await update_triggered_preset(
                user_id=_USER,
                preset_id="preset-1",
                inputs=small_inputs,
                credentials=credentials_from_other_nodes,
            )
    m["setup"].assert_not_awaited()
    m["update"].assert_not_awaited()
    m["prune"].assert_not_awaited()


@pytest.mark.asyncio
async def test_an_update_filled_to_the_bound_registers_no_webhook():
    """Same reserved `payload` slot on the update path: inputs filled right up
    to the bound would leave the trigger node one over it at run time."""
    at_the_bound = {f"field_{i}": i for i in range(MAX_NODE_INPUT_OUTPUT_FETCH)}
    with _update_patches(current=_preset(webhook_id="wh-old")) as m:
        with pytest.raises(InvalidInputError, match="at most"):
            await update_triggered_preset(
                user_id=_USER,
                preset_id="preset-1",
                inputs=at_the_bound,
                credentials={},
            )
    m["setup"].assert_not_awaited()
    m["update"].assert_not_awaited()
    m["prune"].assert_not_awaited()


@pytest.mark.asyncio
async def test_an_update_with_an_oversized_name_registers_no_webhook():
    with _update_patches(current=_preset(webhook_id="wh-old")) as m:
        with pytest.raises(InvalidInputError, match="Preset name is too long"):
            await update_triggered_preset(
                user_id=_USER,
                preset_id="preset-1",
                inputs={"repo": "owner/repo"},
                credentials={},
                name="x" * (db.MAX_PRESET_NAME_LENGTH + 1),
            )
    m["setup"].assert_not_awaited()
    m["update"].assert_not_awaited()


@pytest.mark.asyncio
async def test_an_update_with_an_oversized_payload_registers_no_webhook():
    """A `payload` the caller sends is what is stored, so it is what is measured."""
    with _update_patches(current=_preset(webhook_id="wh-old")) as m:
        with pytest.raises(InvalidInputError, match="inputs hold at most"):
            await update_triggered_preset(
                user_id=_USER,
                preset_id="preset-1",
                inputs={"payload": "x" * (db.MAX_PRESET_INPUT_BYTES + 1)},
                credentials={},
            )
    m["setup"].assert_not_awaited()


@pytest.mark.asyncio
async def test_a_setup_with_an_oversized_payload_registers_no_webhook():
    p_graph, p_creds, p_webhook, p_create, p_expert = _patches(graph=_graph())
    with p_graph, p_creds, p_expert, p_create, p_webhook as webhook_mock:
        with pytest.raises(InvalidInputError, match="inputs hold at most"):
            await _setup(
                trigger_config={"payload": "x" * (db.MAX_PRESET_INPUT_BYTES + 1)}
            )
    webhook_mock.assert_not_awaited()


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
            inputs={"repo": "owner/repo"},
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
    with _not_a_member(), _update_patches(current=_preset(webhook_id="wh-old")) as m:
        await update_triggered_preset(
            user_id=_USER,
            preset_id="preset-1",
            inputs={"repo": "owner/repo"},
            credentials={},
        )
    kwargs = m["setup"].await_args.kwargs
    assert (kwargs["organization_id"], kwargs["team_id"]) == ("own-org", "own-team")


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
            inputs={"repo": "owner/repo"},
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
            inputs={"repo": "owner/repo"},
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
                inputs={"repo": "owner/repo"},
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
                inputs={"repo": "x"},
                credentials={},
            )
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
