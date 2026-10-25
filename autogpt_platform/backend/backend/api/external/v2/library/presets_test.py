"""What the v2 preset routes do beyond the tenant check: webhook upkeep, the
expert a preset runs for, and what a response leaves out."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_mock
from fastapi import HTTPException, Response
from fastapi.exceptions import RequestValidationError
from prisma.enums import APIKeyPermission

from backend.api.features.library.model import LibraryAgentPreset
from backend.copilot.tools.manage_presets import UpdatePresetTool
from backend.data.execution import ExecutionStatus, GraphExecutionMeta
from backend.util.exceptions import ConflictError, ExpertRunPausedError

from ..models import (
    AgentPreset,
    AgentPresetRunRequest,
    AgentPresetUpdateRequest,
    AgentTriggerSetupRequest,
)
from ..tenancy import TenantContext
from . import presets

USER_ID = "user-1"
ORG = "org-1"
SECRET = "whsec-do-not-leak"
PROVIDER_HOOK_ID = "provider-hook-do-not-leak"
TRIGGER_SECRET = "delivery-secret-do-not-leak"


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
        auth=_key(),
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


async def test_setting_up_a_trigger_returns_its_secret_settings_once(
    mocker: pytest_mock.MockFixture,
):
    mocker.patch(
        "backend.data.graph.get_graph", new_callable=AsyncMock, return_value=Mock()
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

    await presets.update_preset(
        request=AgentPresetUpdateRequest(inputs={"constants": {"a": 2}}),
        preset_id="p-1",
        auth=_key(),
    )

    assert update.await_args.kwargs["inputs"] == {
        "constants": {"a": 2},
        "secret_token": TRIGGER_SECRET,
    }


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
    "args, needed",
    [
        ({"preset_id": "p-1", "is_active": True}, [APIKeyPermission.RUN_AGENT]),
        ({"preset_id": "p-1", "is_active": False}, []),
        ({"preset_id": "p-1", "name": "x"}, []),
    ],
)
def test_turning_a_preset_on_over_mcp_needs_the_run_permission(
    args: dict, needed: list[APIKeyPermission]
):
    assert UpdatePresetTool().external_permissions(args) == needed


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
