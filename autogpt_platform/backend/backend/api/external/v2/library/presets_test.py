"""What the v2 preset routes do beyond the tenant check: webhook upkeep, the
expert a preset runs for, and what a response leaves out."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import prisma.models
import pytest
import pytest_mock
from fastapi import HTTPException, Response
from fastapi.exceptions import RequestValidationError
from prisma.enums import APIKeyPermission

from backend.api.features.library.model import LibraryAgentPreset
from backend.copilot.tools.manage_presets import UpdatePresetTool
from backend.data.execution import ExecutionStatus, GraphExecutionMeta
from backend.data.model import CredentialsMetaInput
from backend.util.exceptions import ConflictError, ExpertRunPausedError

from ..models import (
    AgentPreset,
    AgentPresetCreateRequest,
    AgentPresetRunRequest,
    AgentPresetUpdateRequest,
    AgentTriggerSetupRequest,
)
from ..pagination import PageRequest
from ..tenancy import TenantContext
from . import presets

USER_ID = "user-1"
ORG = "org-1"
SECRET = "whsec-do-not-leak"
PROVIDER_HOOK_ID = "provider-hook-do-not-leak"
TRIGGER_SECRET = "delivery-secret-do-not-leak"
RUNNER = [APIKeyPermission.WRITE_LIBRARY, APIKeyPermission.RUN_AGENT]
NEW_CREDENTIALS = {
    "github_credentials": CredentialsMetaInput(
        id="cred-2", provider="github", type="api_key", title="GitHub"
    )
}


@pytest.fixture(autouse=True)
def _no_rate_limit(mocker: pytest_mock.MockFixture) -> None:
    mocker.patch.object(
        presets.graph_exec_limiter, "check", AsyncMock(return_value=None)
    )


@pytest.fixture
def can_pay(mocker: pytest_mock.MockFixture) -> None:
    mocker.patch(
        "backend.api.external.v2.library.helpers.get_credit_model",
        new_callable=AsyncMock,
        return_value=Mock(get_credits=AsyncMock(return_value=100)),
    )


def test_a_preset_response_carries_no_webhook_secret():
    body = AgentPreset.from_internal(_preset(webhook=True)).model_dump_json()

    assert SECRET not in body
    assert PROVIDER_HOOK_ID not in body
    assert "/ingress" in body


async def test_new_trigger_inputs_re_register_the_webhook_with_saved_credentials(
    mocker: pytest_mock.MockFixture,
):
    current = _preset(webhook=True)
    mocker.patch(
        "backend.api.features.library.db.get_preset",
        new_callable=AsyncMock,
        return_value=current,
    )
    update = mocker.patch(
        "backend.api.features.library.triggers.update_triggered_preset",
        new_callable=AsyncMock,
        return_value=current,
    )

    await presets.update_preset(
        request=AgentPresetUpdateRequest(inputs={"repo": "a/b"}),
        preset_id=current.id,
        auth=_key(RUNNER),
    )

    assert update.await_args.kwargs["inputs"] == {"repo": "a/b"}
    assert update.await_args.kwargs["credentials"] == current.credentials


async def test_deleting_a_trigger_cleans_up_its_webhook(
    mocker: pytest_mock.MockFixture,
):
    mocker.patch(
        "backend.api.features.library.db.get_preset",
        new_callable=AsyncMock,
        return_value=_preset(webhook=True),
    )
    cleanup = mocker.patch(
        "backend.api.features.library.triggers.delete_preset_with_webhook_cleanup",
        new_callable=AsyncMock,
    )

    await presets.delete_preset(preset_id="p-1", auth=_key())

    cleanup.assert_awaited_once_with(user_id=USER_ID, preset_id="p-1")


async def test_a_run_from_an_expert_preset_counts_against_that_expert(
    mocker: pytest_mock.MockFixture, can_pay: None
):
    _stub_preset_and_graph(mocker, _preset(expert_id="expert-1"))
    add = mocker.patch(
        "backend.executor.utils.add_graph_execution",
        new_callable=AsyncMock,
        return_value=_run(),
    )

    await _execute()

    assert add.await_args.kwargs["expert_id"] == "expert-1"
    assert add.await_args.kwargs["preset_id"] == "p-1"


async def test_a_paused_expert_s_preset_is_a_conflict(
    mocker: pytest_mock.MockFixture, can_pay: None
):
    _stub_preset_and_graph(mocker, _preset(expert_id="expert-1"))
    mocker.patch(
        "backend.executor.utils.add_graph_execution",
        new_callable=AsyncMock,
        side_effect=ExpertRunPausedError("paused", "expert-1"),
    )

    with pytest.raises(ConflictError):
        await _execute()


async def test_a_run_missing_a_required_input_is_refused(
    mocker: pytest_mock.MockFixture, can_pay: None
):
    _stub_preset_and_graph(
        mocker,
        _preset(),
        input_schema={"properties": {"topic": {}}, "required": ["topic"]},
    )
    add = mocker.patch(
        "backend.executor.utils.add_graph_execution", new_callable=AsyncMock
    )

    with pytest.raises(RequestValidationError):
        await _execute()
    add.assert_not_awaited()


async def test_a_trigger_s_settings_are_not_checked_as_graph_inputs(
    mocker: pytest_mock.MockFixture, can_pay: None
):
    """A trigger's inputs configure its block; the graph's inputs come from
    each event."""
    _stub_preset_and_graph(
        mocker,
        _trigger(),
        input_schema={"properties": {"topic": {}}, "required": ["topic"]},
    )
    add = mocker.patch(
        "backend.executor.utils.add_graph_execution",
        new_callable=AsyncMock,
        return_value=_run(),
    )

    await _execute()

    assert add.await_args.kwargs["inputs"]["secret_token"] == TRIGGER_SECRET


@pytest.mark.parametrize("route", ["create", "setup-trigger"])
async def test_a_preset_for_a_graph_a_hired_expert_installed_is_that_expert_s(
    mocker: pytest_mock.MockFixture, route: str
):
    """As on the internal routes, so its runs meet the expert's budget and pause."""
    mocker.patch(
        "backend.api.features.experts.experts_db.resolve_expert_for_graph",
        new_callable=AsyncMock,
        return_value="expert-1",
    )
    mocker.patch(
        "backend.data.graph.get_graph",
        new_callable=AsyncMock,
        return_value=Mock(input_schema={"properties": {}}, webhook_input_node=None),
    )
    save = mocker.patch(
        (
            "backend.api.features.library.db.create_preset"
            if route == "create"
            else "backend.api.features.library.triggers.setup_triggered_preset"
        ),
        new_callable=AsyncMock,
        return_value=_preset(),
    )

    if route == "create":
        await presets.create_preset(
            request=AgentPresetCreateRequest(
                name="Preset", graph_id="graph-1", graph_version=1
            ),
            auth=_key(),
        )
    else:
        await presets.setup_trigger(
            request=AgentTriggerSetupRequest(
                name="On delivery",
                graph_id="graph-1",
                graph_version=1,
                trigger_config={},
            ),
            auth=_key(),
        )

    assert save.await_args.kwargs["expert_id"] == "expert-1"


def test_the_hidden_settings_are_every_trigger_block_s_secret_inputs():
    """Read from the blocks, so a new trigger's secret is hidden without a list."""
    assert {"secret_token", "signing_secret"} <= presets.secret_trigger_inputs()


async def test_a_read_withholds_a_trigger_s_secret_settings(
    mocker: pytest_mock.MockFixture,
):
    mocker.patch(
        "backend.api.features.library.db.get_preset",
        new_callable=AsyncMock,
        return_value=_trigger(),
    )

    read = await presets.get_preset(preset_id="p-1", auth=_key())

    assert TRIGGER_SECRET not in read.model_dump_json()
    assert read.inputs == {"constants": {"a": 1}}
    assert read.hidden_inputs == ["secret_token"]


async def test_a_listing_withholds_a_trigger_s_secret_settings(
    mocker: pytest_mock.MockFixture,
):
    mocker.patch(
        "backend.api.features.library.db.list_presets",
        new_callable=AsyncMock,
        return_value=Mock(presets=[_trigger()], pagination=Mock(total_items=1)),
    )

    listed = await presets.list_presets(
        graph_id=None, page=PageRequest(limit=20), auth=_key()
    )

    assert TRIGGER_SECRET not in listed.model_dump_json()
    assert listed.items[0].hidden_inputs == ["secret_token"]


async def test_setting_up_a_trigger_returns_its_secret_settings_once(
    mocker: pytest_mock.MockFixture,
):
    mocker.patch(
        "backend.data.graph.get_graph", new_callable=AsyncMock, return_value=Mock()
    )
    mocker.patch(
        "backend.api.features.experts.experts_db.resolve_expert_for_graph",
        new_callable=AsyncMock,
        return_value=None,
    )
    mocker.patch(
        "backend.api.features.library.triggers.setup_triggered_preset",
        new_callable=AsyncMock,
        return_value=_trigger(),
    )

    created = await presets.setup_trigger(
        request=AgentTriggerSetupRequest(
            name="On delivery", graph_id="graph-1", graph_version=1, trigger_config={}
        ),
        auth=_key(),
    )

    assert created.inputs["secret_token"] == TRIGGER_SECRET
    assert created.hidden_inputs == []


async def test_an_update_that_leaves_out_a_hidden_setting_keeps_it(
    mocker: pytest_mock.MockFixture,
):
    """A client sends back what it read; that must not switch off the check."""
    mocker.patch(
        "backend.api.features.library.db.get_preset",
        new_callable=AsyncMock,
        return_value=_trigger(),
    )
    update = mocker.patch(
        "backend.api.features.library.triggers.update_triggered_preset",
        new_callable=AsyncMock,
        return_value=_trigger(),
    )

    updated = await presets.update_preset(
        request=AgentPresetUpdateRequest(inputs={"constants": {"a": 2}}),
        preset_id="p-1",
        auth=_key(RUNNER),
    )

    assert update.await_args.kwargs["inputs"] == {
        "constants": {"a": 2},
        "secret_token": TRIGGER_SECRET,
    }
    assert TRIGGER_SECRET not in updated.model_dump_json()


@pytest.mark.parametrize("scopes, allowed", [([], False), (["RUN_AGENT"], True)])
async def test_turning_a_preset_on_needs_the_run_permission(
    mocker: pytest_mock.MockFixture, scopes: list[str], allowed: bool
):
    """An active trigger starts runs, which spend credits."""
    mocker.patch(
        "backend.api.features.library.db.get_preset",
        new_callable=AsyncMock,
        return_value=_trigger(),
    )
    update = mocker.patch(
        "backend.api.features.library.triggers.update_triggered_preset",
        new_callable=AsyncMock,
        return_value=_trigger(),
    )
    request = AgentPresetUpdateRequest(is_active=True)
    auth = _key([APIKeyPermission.WRITE_LIBRARY, *map(APIKeyPermission, scopes)])

    if allowed:
        await presets.update_preset(request=request, preset_id="p-1", auth=auth)
        update.assert_awaited_once()
    else:
        with pytest.raises(HTTPException) as refused:
            await presets.update_preset(request=request, preset_id="p-1", auth=auth)
        assert refused.value.status_code == 403
        update.assert_not_awaited()


@pytest.mark.parametrize(
    "is_active, trigger_graph, request_active, refused",
    [
        (True, True, None, True),
        (True, True, False, False),
        (True, False, None, False),
        (False, True, None, False),
    ],
    ids=["arms-a-trigger", "turns-it-off", "not-a-trigger", "stays-off"],
)
async def test_new_inputs_on_a_trigger_that_stays_on_need_the_run_permission(
    mocker: pytest_mock.MockFixture,
    is_active: bool,
    trigger_graph: bool,
    request_active: bool | None,
    refused: bool,
):
    """New inputs register its webhook, and an active trigger starts runs."""
    mocker.patch(
        "backend.api.features.library.db.get_preset",
        new_callable=AsyncMock,
        return_value=_preset().model_copy(update={"is_active": is_active}),
    )
    mocker.patch(
        "backend.data.graph.get_graph",
        new_callable=AsyncMock,
        return_value=Mock(webhook_input_node=Mock() if trigger_graph else None),
    )
    update = mocker.patch(
        "backend.api.features.library.triggers.update_triggered_preset",
        new_callable=AsyncMock,
        return_value=_preset(),
    )
    request = AgentPresetUpdateRequest(inputs={"repo": "a/b"}, is_active=request_active)
    writer = _key([APIKeyPermission.WRITE_LIBRARY])

    if not refused:
        await presets.update_preset(request=request, preset_id="p-1", auth=writer)
        update.assert_awaited_once()
        return
    with pytest.raises(HTTPException) as refusal:
        await presets.update_preset(request=request, preset_id="p-1", auth=writer)
    assert refusal.value.status_code == 403
    update.assert_not_awaited()


@pytest.mark.parametrize(
    "trigger_graph, is_active, refused",
    [(True, True, True), (True, False, False), (False, True, False)],
    ids=["active-trigger", "inactive-trigger", "not-a-trigger"],
)
async def test_creating_an_active_trigger_preset_needs_the_run_permission(
    mocker: pytest_mock.MockFixture,
    trigger_graph: bool,
    is_active: bool,
    refused: bool,
):
    """New inputs would then arm it, and turning it on needs the permission."""
    mocker.patch(
        "backend.data.graph.get_graph",
        new_callable=AsyncMock,
        return_value=Mock(
            input_schema={"properties": {}},
            webhook_input_node=Mock() if trigger_graph else None,
        ),
    )
    mocker.patch(
        "backend.api.features.experts.experts_db.resolve_expert_for_graph",
        new_callable=AsyncMock,
        return_value=None,
    )
    create = mocker.patch(
        "backend.api.features.library.db.create_preset",
        new_callable=AsyncMock,
        return_value=_preset(),
    )
    request = AgentPresetCreateRequest(
        name="Preset", graph_id="graph-1", graph_version=1, is_active=is_active
    )
    writer = _key([APIKeyPermission.WRITE_LIBRARY])

    if not refused:
        await presets.create_preset(request=request, auth=writer)
        create.assert_awaited_once()
        return
    with pytest.raises(HTTPException) as refusal:
        await presets.create_preset(request=request, auth=writer)
    assert refusal.value.status_code == 403
    create.assert_not_awaited()


@pytest.mark.parametrize(
    "args, needed",
    [
        ({"preset_id": "p-1", "is_active": True}, [APIKeyPermission.RUN_AGENT]),
        ({"preset_id": "p-1", "is_active": False}, []),
        ({"preset_id": "p-1", "name": "x"}, []),
        # New inputs re-register a trigger's webhook, and the tool can't see
        # the graph before it runs.
        ({"preset_id": "p-1", "inputs": {"a": 1}}, [APIKeyPermission.RUN_AGENT]),
        ({"preset_id": "p-1", "inputs": {"a": 1}, "is_active": False}, []),
    ],
)
def test_turning_a_preset_on_over_mcp_needs_the_run_permission(
    args: dict, needed: list[APIKeyPermission]
):
    assert UpdatePresetTool().external_permissions(args) == needed


async def test_new_inputs_on_a_preset_without_credentials_are_saved(
    mocker: pytest_mock.MockFixture, stored: AsyncMock
):
    """Its saved credentials, `{}`, go along with the new inputs."""
    mocker.patch(
        "backend.api.features.library.db.get_preset",
        new_callable=AsyncMock,
        return_value=_preset(),
    )
    mocker.patch(
        "backend.api.features.library.triggers.get_graph",
        new_callable=AsyncMock,
        return_value=Mock(webhook_input_node=None),
    )

    await presets.update_preset(
        request=AgentPresetUpdateRequest(inputs={"topic": "x"}),
        preset_id="p-1",
        auth=_key(RUNNER),
    )

    rows = stored.update.await_args.kwargs["data"]["InputPresets"]["create"]
    assert [row["name"] for row in rows] == ["topic"]


async def test_new_inputs_on_a_trigger_without_credentials_move_it_to_its_new_webhook(
    mocker: pytest_mock.MockFixture, stored: AsyncMock
):
    """The new webhook is registered before the inputs are saved, so a refused
    save would leave it attached to nothing."""
    mocker.patch(
        "backend.api.features.library.db.get_preset",
        new_callable=AsyncMock,
        return_value=_trigger(),
    )
    mocker.patch(
        "backend.api.features.library.triggers.get_graph",
        new_callable=AsyncMock,
        return_value=Mock(
            webhook_input_node=Mock(id="node-1"), organization_id=ORG, team_id=None
        ),
    )
    mocker.patch(
        "backend.api.features.library.triggers.resolve_member_tenancy",
        new_callable=AsyncMock,
        return_value=(ORG, None),
    )
    mocker.patch(
        "backend.api.features.library.triggers.make_node_credentials_input_map",
        return_value={},
    )
    mocker.patch(
        "backend.api.features.library.triggers.setup_webhook_for_block",
        new_callable=AsyncMock,
        return_value=(Mock(id="hook-2"), None),
    )
    attach = mocker.patch(
        "backend.api.features.library.db.set_preset_webhook",
        new_callable=AsyncMock,
        return_value=_trigger(),
    )
    prune = mocker.patch(
        "backend.api.features.library.triggers._prune_dangling_webhook",
        new_callable=AsyncMock,
    )

    await presets.update_preset(
        request=AgentPresetUpdateRequest(inputs={"constants": {"a": 2}}),
        preset_id="p-1",
        auth=_key(RUNNER),
    )

    attach.assert_awaited_once_with(USER_ID, "p-1", "hook-2")
    prune.assert_awaited_once_with(USER_ID, "hook-1")


async def test_new_credentials_alone_are_saved_with_the_saved_inputs(
    mocker: pytest_mock.MockFixture, stored: AsyncMock
):
    mocker.patch(
        "backend.api.features.library.db.get_preset",
        new_callable=AsyncMock,
        return_value=_preset(inputs={"topic": "x"}),
    )
    mocker.patch(
        "backend.api.features.library.triggers.get_graph",
        new_callable=AsyncMock,
        return_value=Mock(webhook_input_node=None),
    )

    await presets.update_preset(
        request=AgentPresetUpdateRequest(credentials_inputs=NEW_CREDENTIALS),
        preset_id="p-1",
        auth=_key(RUNNER),
    )

    rows = stored.update.await_args.kwargs["data"]["InputPresets"]["create"]
    assert sorted(row["name"] for row in rows) == ["github_credentials", "topic"]


async def test_new_credentials_alone_move_a_trigger_keeping_its_secret_settings(
    mocker: pytest_mock.MockFixture, stored: AsyncMock
):
    """Reads leave the secret settings out, so they come from the saved preset."""
    mocker.patch(
        "backend.api.features.library.db.get_preset",
        new_callable=AsyncMock,
        return_value=_trigger(),
    )
    mocker.patch(
        "backend.api.features.library.triggers.get_graph",
        new_callable=AsyncMock,
        return_value=Mock(
            webhook_input_node=Mock(id="node-1"), organization_id=ORG, team_id=None
        ),
    )
    mocker.patch(
        "backend.api.features.library.triggers.resolve_member_tenancy",
        new_callable=AsyncMock,
        return_value=(ORG, None),
    )
    mocker.patch(
        "backend.api.features.library.triggers.make_node_credentials_input_map",
        return_value={},
    )
    register = mocker.patch(
        "backend.api.features.library.triggers.setup_webhook_for_block",
        new_callable=AsyncMock,
        return_value=(Mock(id="hook-2"), None),
    )
    mocker.patch(
        "backend.api.features.library.db.set_preset_webhook",
        new_callable=AsyncMock,
        return_value=_trigger(),
    )
    mocker.patch(
        "backend.api.features.library.triggers._prune_dangling_webhook",
        new_callable=AsyncMock,
    )

    await presets.update_preset(
        request=AgentPresetUpdateRequest(credentials_inputs=NEW_CREDENTIALS),
        preset_id="p-1",
        auth=_key(RUNNER),
    )

    assert register.await_args.kwargs["trigger_config"]["secret_token"] == (
        TRIGGER_SECRET
    )
    rows = stored.update.await_args.kwargs["data"]["InputPresets"]["create"]
    assert sorted(row["name"] for row in rows) == [
        "constants",
        "github_credentials",
        "secret_token",
    ]


@pytest.fixture
def stored(mocker: pytest_mock.MockFixture) -> AsyncMock:
    """`db.update_preset` runs for real; only its Prisma calls are stubbed."""

    @asynccontextmanager
    async def no_transaction():
        yield None

    client = AsyncMock()
    mocker.patch("backend.api.features.library.db.transaction", no_transaction)
    mocker.patch.object(prisma.models.AgentPreset, "prisma", return_value=client)
    mocker.patch.object(
        prisma.models.AgentNodeExecutionInputOutput, "prisma", return_value=AsyncMock()
    )
    mocker.patch.object(LibraryAgentPreset, "from_db", return_value=_preset())
    return client


def _trigger() -> LibraryAgentPreset:
    return _preset(
        webhook=True,
        inputs={"secret_token": TRIGGER_SECRET, "constants": {"a": 1}},
    )


def _stub_preset_and_graph(
    mocker: pytest_mock.MockFixture,
    preset: LibraryAgentPreset,
    input_schema: dict | None = None,
) -> None:
    mocker.patch(
        "backend.api.features.library.db.get_preset",
        new_callable=AsyncMock,
        return_value=preset,
    )
    mocker.patch(
        "backend.data.graph.get_graph",
        new_callable=AsyncMock,
        return_value=Mock(input_schema=input_schema or {"properties": {}}),
    )


async def _execute():
    return await presets.execute_preset(
        response=Response(),
        preset_id="p-1",
        request=AgentPresetRunRequest(),
        auth=_key(),
    )


def _preset(
    webhook: bool = False, expert_id: str | None = None, inputs: dict | None = None
) -> LibraryAgentPreset:
    now = datetime.now(tz=timezone.utc)
    return LibraryAgentPreset.model_validate(
        {
            "id": "p-1",
            "user_id": USER_ID,
            "graph_id": "graph-1",
            "graph_version": 1,
            "name": "Preset",
            "description": "",
            "inputs": inputs or {},
            "credentials": {},
            "created_at": now,
            "updated_at": now,
            "organization_id": ORG,
            "expert_id": expert_id,
            "webhook_id": "hook-1" if webhook else None,
            "webhook": (
                {
                    "id": "hook-1",
                    "user_id": USER_ID,
                    "provider": "generic_webhook",
                    "credentials_id": "",
                    "webhook_type": "plain",
                    "resource": "",
                    "events": [],
                    "secret": SECRET,
                    "provider_webhook_id": PROVIDER_HOOK_ID,
                }
                if webhook
                else None
            ),
        }
    )


def _run() -> GraphExecutionMeta:
    return GraphExecutionMeta.model_construct(
        id="run-1",
        user_id=USER_ID,
        graph_id="graph-1",
        graph_version=1,
        preset_id="p-1",
        status=ExecutionStatus.QUEUED,
        started_at=None,
        ended_at=None,
        inputs={},
        credential_inputs=None,
        is_shared=False,
        share_token=None,
        stats=None,
        organization_id=ORG,
        team_id=None,
    )


def _key(scopes: list[APIKeyPermission] | None = None) -> TenantContext:
    return TenantContext(
        user_id=USER_ID,
        scopes=scopes or [],
        type="api_key",
        organization_id=ORG,
    )
