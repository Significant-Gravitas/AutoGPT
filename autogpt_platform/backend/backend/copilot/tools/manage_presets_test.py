"""Tests for the preset-management tools (list / update / delete)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.manage_presets import (
    DeletePresetTool,
    ListPresetsTool,
    PresetDeletedResponse,
    PresetListResponse,
    PresetUpdatedResponse,
    UpdatePresetTool,
)
from backend.copilot.tools.models import ErrorResponse
from backend.util.exceptions import (
    InvalidInputError,
    NotFoundError,
    WebhookRegistrationError,
)

from ._test_data import make_session

_USER = "test-user-presets"
_PATH = "backend.copilot.tools.manage_presets"


@pytest.fixture
def session():
    return make_session(_USER)


def _preset(*, id="preset-1", name="My Preset", is_active=True, webhook=True):
    preset = MagicMock()
    preset.id = id
    preset.name = name
    preset.description = "desc"
    preset.graph_id = "graph-1"
    preset.graph_version = 1
    preset.is_active = is_active
    preset.webhook_id = "wh-1" if webhook else None
    preset.inputs = {"repo": "owner/repo"}
    preset.credentials = {}
    preset.webhook = (
        MagicMock(url="https://x/ingress", provider="github") if webhook else None
    )
    return preset


# ---- list_presets ----


@pytest.mark.asyncio
async def test_list_no_auth(session):
    result = await ListPresetsTool()._execute(user_id=None, session=session)
    assert isinstance(result, ErrorResponse)
    assert result.error == "auth_required"


@pytest.mark.asyncio
async def test_list_empty(session):
    ldb = MagicMock()
    ldb.list_presets = AsyncMock(
        return_value=MagicMock(presets=[], pagination=MagicMock(total_items=0))
    )
    with patch(f"{_PATH}.library_db", return_value=ldb):
        result = await ListPresetsTool()._execute(user_id=_USER, session=session)
    assert isinstance(result, PresetListResponse)
    assert result.presets == []


@pytest.mark.asyncio
async def test_list_populated_with_graph_filter(session):
    ldb = MagicMock()
    ldb.list_presets = AsyncMock(
        return_value=MagicMock(presets=[_preset()], pagination=MagicMock(total_items=1))
    )
    with patch(f"{_PATH}.library_db", return_value=ldb):
        result = await ListPresetsTool()._execute(
            user_id=_USER, session=session, graph_id="graph-1"
        )
    assert isinstance(result, PresetListResponse)
    assert len(result.presets) == 1
    assert result.total_count == 1
    assert result.presets[0].webhook_url == "https://x/ingress"
    assert ldb.list_presets.await_args.kwargs["graph_id"] == "graph-1"


@pytest.mark.asyncio
async def test_list_truncation_hint_when_more_than_one_page(session):
    # 1 preset returned but 247 total -> message must flag truncation + total.
    ldb = MagicMock()
    ldb.list_presets = AsyncMock(
        return_value=MagicMock(
            presets=[_preset()], pagination=MagicMock(total_items=247)
        )
    )
    with patch(f"{_PATH}.library_db", return_value=ldb):
        result = await ListPresetsTool()._execute(user_id=_USER, session=session)
    assert isinstance(result, PresetListResponse)
    assert result.total_count == 247
    assert "247" in result.message and "first 1" in result.message


@pytest.mark.asyncio
async def test_list_resolves_library_agent_id(session):
    ldb = MagicMock()
    ldb.get_library_agent = AsyncMock(return_value=MagicMock(graph_id="graph-xyz"))
    ldb.list_presets = AsyncMock(
        return_value=MagicMock(presets=[], pagination=MagicMock(total_items=0))
    )
    with patch(f"{_PATH}.library_db", return_value=ldb):
        await ListPresetsTool()._execute(
            user_id=_USER, session=session, library_agent_id="lib-1"
        )
    assert ldb.list_presets.await_args.kwargs["graph_id"] == "graph-xyz"


# ---- update_preset ----


@pytest.mark.asyncio
async def test_update_missing_preset_id(session):
    result = await UpdatePresetTool()._execute(user_id=_USER, session=session)
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_argument"


@pytest.mark.asyncio
async def test_update_rename_skips_preset_fetch(session):
    tdb = MagicMock()
    tdb.update_triggered_preset = AsyncMock(return_value=_preset(name="Renamed"))
    ldb = MagicMock()
    ldb.get_preset = AsyncMock()
    with patch(f"{_PATH}.triggers_db", return_value=tdb), patch(
        f"{_PATH}.library_db", return_value=ldb
    ):
        result = await UpdatePresetTool()._execute(
            user_id=_USER, session=session, preset_id="preset-1", name="Renamed"
        )
    assert isinstance(result, PresetUpdatedResponse)
    assert result.name == "Renamed"
    ldb.get_preset.assert_not_awaited()
    assert tdb.update_triggered_preset.await_args.kwargs["inputs"] is None


@pytest.mark.asyncio
async def test_update_pause(session):
    tdb = MagicMock()
    tdb.update_triggered_preset = AsyncMock(return_value=_preset(is_active=False))
    with patch(f"{_PATH}.triggers_db", return_value=tdb), patch(
        f"{_PATH}.library_db", return_value=MagicMock()
    ):
        result = await UpdatePresetTool()._execute(
            user_id=_USER, session=session, preset_id="preset-1", is_active=False
        )
    assert isinstance(result, PresetUpdatedResponse)
    assert result.is_active is False
    assert tdb.update_triggered_preset.await_args.kwargs["is_active"] is False


@pytest.mark.asyncio
async def test_update_reconfigure_merges_and_reuses_credentials(session):
    current = _preset()
    current.inputs = {"repo": "owner/repo", "events": ["push"]}
    current.credentials = {"github": MagicMock()}
    ldb = MagicMock()
    ldb.get_preset = AsyncMock(return_value=current)
    tdb = MagicMock()
    tdb.update_triggered_preset = AsyncMock(return_value=_preset())
    with patch(f"{_PATH}.library_db", return_value=ldb), patch(
        f"{_PATH}.triggers_db", return_value=tdb
    ):
        await UpdatePresetTool()._execute(
            user_id=_USER,
            session=session,
            preset_id="preset-1",
            inputs={"events": ["push", "pull_request"]},
        )
    kwargs = tdb.update_triggered_preset.await_args.kwargs
    assert kwargs["inputs"] == {
        "repo": "owner/repo",
        "events": ["push", "pull_request"],
    }
    assert kwargs["credentials"] == current.credentials


@pytest.mark.asyncio
async def test_update_reconfigure_preset_not_found(session):
    ldb = MagicMock()
    ldb.get_preset = AsyncMock(return_value=None)
    with patch(f"{_PATH}.library_db", return_value=ldb), patch(
        f"{_PATH}.triggers_db", return_value=MagicMock()
    ):
        result = await UpdatePresetTool()._execute(
            user_id=_USER, session=session, preset_id="missing", inputs={"x": 1}
        )
    assert isinstance(result, ErrorResponse)
    assert result.error == "preset_not_found"


@pytest.mark.asyncio
async def test_update_not_found_from_shared_fn(session):
    tdb = MagicMock()
    tdb.update_triggered_preset = AsyncMock(side_effect=NotFoundError("gone"))
    with patch(f"{_PATH}.triggers_db", return_value=tdb), patch(
        f"{_PATH}.library_db", return_value=MagicMock()
    ):
        result = await UpdatePresetTool()._execute(
            user_id=_USER, session=session, preset_id="missing", name="X"
        )
    assert isinstance(result, ErrorResponse)
    assert result.error == "preset_not_found"


@pytest.mark.asyncio
async def test_update_webhook_rejected(session):
    ldb = MagicMock()
    ldb.get_preset = AsyncMock(return_value=_preset())
    tdb = MagicMock()
    tdb.update_triggered_preset = AsyncMock(side_effect=InvalidInputError("no events"))
    with patch(f"{_PATH}.library_db", return_value=ldb), patch(
        f"{_PATH}.triggers_db", return_value=tdb
    ):
        result = await UpdatePresetTool()._execute(
            user_id=_USER,
            session=session,
            preset_id="preset-1",
            inputs={"events": []},
        )
    assert isinstance(result, ErrorResponse)
    assert result.error == "preset_update_failed"
    assert "no events" in result.message


# ---- delete_preset ----


@pytest.mark.asyncio
async def test_delete_missing_preset_id(session):
    result = await DeletePresetTool()._execute(user_id=_USER, session=session)
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_argument"


@pytest.mark.asyncio
async def test_delete_not_found(session):
    ldb = MagicMock()
    ldb.get_preset = AsyncMock(return_value=None)
    with patch(f"{_PATH}.library_db", return_value=ldb), patch(
        f"{_PATH}.triggers_db", return_value=MagicMock()
    ):
        result = await DeletePresetTool()._execute(
            user_id=_USER, session=session, preset_id="missing"
        )
    assert isinstance(result, ErrorResponse)
    assert result.error == "preset_not_found"


@pytest.mark.asyncio
async def test_delete_success(session):
    ldb = MagicMock()
    ldb.get_preset = AsyncMock(return_value=_preset(name="ToDelete"))
    tdb = MagicMock()
    tdb.delete_preset_with_webhook_cleanup = AsyncMock()
    with patch(f"{_PATH}.library_db", return_value=ldb), patch(
        f"{_PATH}.triggers_db", return_value=tdb
    ):
        result = await DeletePresetTool()._execute(
            user_id=_USER, session=session, preset_id="preset-1"
        )
    assert isinstance(result, PresetDeletedResponse)
    assert result.name == "ToDelete"
    tdb.delete_preset_with_webhook_cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_no_auth(session):
    result = await UpdatePresetTool()._execute(user_id=None, session=session)
    assert isinstance(result, ErrorResponse)
    assert result.error == "auth_required"


@pytest.mark.asyncio
async def test_delete_no_auth(session):
    result = await DeletePresetTool()._execute(user_id=None, session=session)
    assert isinstance(result, ErrorResponse)
    assert result.error == "auth_required"


@pytest.mark.asyncio
async def test_update_webhook_registration_error(session):
    """A provider webhook failure during reconfigure surfaces as a clean
    preset_update_failed, not an unhandled tool error."""
    ldb = MagicMock()
    ldb.get_preset = AsyncMock(return_value=_preset())
    tdb = MagicMock()
    tdb.update_triggered_preset = AsyncMock(
        side_effect=WebhookRegistrationError("provider refused")
    )
    with patch(f"{_PATH}.library_db", return_value=ldb), patch(
        f"{_PATH}.triggers_db", return_value=tdb
    ):
        result = await UpdatePresetTool()._execute(
            user_id=_USER, session=session, preset_id="preset-1", inputs={"repo": "x"}
        )
    assert isinstance(result, ErrorResponse)
    assert result.error == "preset_update_failed"
    assert "provider refused" in result.message
