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


def _preset(
    *,
    id="preset-1",
    name="My Preset",
    is_active=True,
    webhook=True,
    expert_id=None,
):
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
    preset.expert_id = expert_id
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
    first_page = [_preset(id=f"preset-{i}") for i in range(100)]
    ldb = MagicMock()
    ldb.list_presets = AsyncMock(
        side_effect=[
            MagicMock(
                presets=first_page,
                pagination=MagicMock(total_items=101),
            ),
            MagicMock(
                presets=[_preset(id="preset-100")],
                pagination=MagicMock(total_items=101),
            ),
        ]
    )
    with patch(f"{_PATH}.library_db", return_value=ldb):
        result = await ListPresetsTool()._execute(user_id=_USER, session=session)
    assert isinstance(result, PresetListResponse)
    assert result.total_count == 101
    assert len(result.presets) == 100
    assert "101" in result.message and "1-100" in result.message


@pytest.mark.asyncio
async def test_list_second_page_is_exactly_scoped_to_current_expert():
    session = make_session(_USER, expert_id="expert-a")
    first_page = [
        *[_preset(id=f"a-{i}", expert_id="expert-a") for i in range(60)],
        *[_preset(id=f"b-{i}", expert_id="expert-b") for i in range(40)],
    ]
    second_page = [
        *[_preset(id=f"a-{i}", expert_id="expert-a") for i in range(60, 110)],
        *[_preset(id=f"b-{i}", expert_id="expert-b") for i in range(40, 80)],
    ]
    ldb = MagicMock()
    ldb.list_presets = AsyncMock(
        side_effect=[
            MagicMock(presets=first_page, pagination=MagicMock(total_items=190)),
            MagicMock(presets=second_page, pagination=MagicMock(total_items=190)),
        ]
    )

    with patch(f"{_PATH}.library_db", return_value=ldb):
        result = await ListPresetsTool()._execute(
            user_id=_USER, session=session, page=2, page_size=100
        )

    assert isinstance(result, PresetListResponse)
    assert result.total_count == 110
    assert result.page == 2
    assert result.page_size == 100
    assert [preset.id for preset in result.presets] == [
        f"a-{i}" for i in range(100, 110)
    ]
    assert all(not preset.id.startswith("b-") for preset in result.presets)


@pytest.mark.parametrize(("page", "page_size"), [(0, 100), (1, 0), (1, 101)])
@pytest.mark.asyncio
async def test_list_rejects_unsafe_pagination(session, page, page_size):
    result = await ListPresetsTool()._execute(
        user_id=_USER, session=session, page=page, page_size=page_size
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "invalid_pagination"


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


@pytest.mark.parametrize(
    ("session_expert_id", "expected_id"),
    [(None, "autopilot"), ("expert-a", "preset-a"), ("expert-b", "preset-b")],
)
@pytest.mark.asyncio
async def test_list_only_returns_current_persona_scope(session_expert_id, expected_id):
    scoped_session = make_session(_USER, expert_id=session_expert_id)
    ldb = MagicMock()
    ldb.list_presets = AsyncMock(
        return_value=MagicMock(
            presets=[
                _preset(id="autopilot", expert_id=None),
                _preset(id="preset-a", expert_id="expert-a"),
                _preset(id="preset-b", expert_id="expert-b"),
            ],
            pagination=MagicMock(total_items=3),
        )
    )

    with patch(f"{_PATH}.library_db", return_value=ldb):
        result = await ListPresetsTool()._execute(user_id=_USER, session=scoped_session)

    assert isinstance(result, PresetListResponse)
    assert result.total_count == 1
    assert [preset.id for preset in result.presets] == [expected_id]


# ---- update_preset ----


@pytest.mark.asyncio
async def test_update_missing_preset_id(session):
    result = await UpdatePresetTool()._execute(user_id=_USER, session=session)
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_argument"


@pytest.mark.asyncio
async def test_update_rename_validates_preset_scope(session):
    tdb = MagicMock()
    tdb.update_triggered_preset = AsyncMock(return_value=_preset(name="Renamed"))
    ldb = MagicMock()
    ldb.get_preset = AsyncMock(return_value=_preset())
    with (
        patch(f"{_PATH}.triggers_db", return_value=tdb),
        patch(f"{_PATH}.library_db", return_value=ldb),
    ):
        result = await UpdatePresetTool()._execute(
            user_id=_USER, session=session, preset_id="preset-1", name="Renamed"
        )
    assert isinstance(result, PresetUpdatedResponse)
    assert result.name == "Renamed"
    ldb.get_preset.assert_awaited_once_with(user_id=_USER, preset_id="preset-1")
    assert tdb.update_triggered_preset.await_args.kwargs["inputs"] is None


@pytest.mark.asyncio
async def test_update_pause(session):
    tdb = MagicMock()
    tdb.update_triggered_preset = AsyncMock(return_value=_preset(is_active=False))
    ldb = MagicMock()
    ldb.get_preset = AsyncMock(return_value=_preset())
    with (
        patch(f"{_PATH}.triggers_db", return_value=tdb),
        patch(f"{_PATH}.library_db", return_value=ldb),
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
    with (
        patch(f"{_PATH}.library_db", return_value=ldb),
        patch(f"{_PATH}.triggers_db", return_value=tdb),
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
    with (
        patch(f"{_PATH}.library_db", return_value=ldb),
        patch(f"{_PATH}.triggers_db", return_value=MagicMock()),
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
    ldb = MagicMock()
    ldb.get_preset = AsyncMock(return_value=_preset())
    with (
        patch(f"{_PATH}.triggers_db", return_value=tdb),
        patch(f"{_PATH}.library_db", return_value=ldb),
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
    with (
        patch(f"{_PATH}.library_db", return_value=ldb),
        patch(f"{_PATH}.triggers_db", return_value=tdb),
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


@pytest.mark.parametrize(
    ("session_expert_id", "target_expert_id"),
    [
        (None, "expert-a"),
        ("expert-a", None),
        ("expert-a", "expert-b"),
        ("expert-b", "expert-a"),
    ],
)
@pytest.mark.asyncio
async def test_update_refuses_cross_persona_preset(session_expert_id, target_expert_id):
    scoped_session = make_session(_USER, expert_id=session_expert_id)
    ldb = MagicMock()
    ldb.get_preset = AsyncMock(return_value=_preset(expert_id=target_expert_id))
    tdb = MagicMock()
    tdb.update_triggered_preset = AsyncMock()

    with (
        patch(f"{_PATH}.library_db", return_value=ldb),
        patch(f"{_PATH}.triggers_db", return_value=tdb),
    ):
        result = await UpdatePresetTool()._execute(
            user_id=_USER,
            session=scoped_session,
            preset_id="foreign-preset",
            name="Stolen",
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "preset_not_found"
    tdb.update_triggered_preset.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_allows_same_expert_preset():
    scoped_session = make_session(_USER, expert_id="expert-a")
    ldb = MagicMock()
    ldb.get_preset = AsyncMock(return_value=_preset(expert_id="expert-a"))
    tdb = MagicMock()
    tdb.update_triggered_preset = AsyncMock(
        return_value=_preset(name="Renamed", expert_id="expert-a")
    )

    with (
        patch(f"{_PATH}.library_db", return_value=ldb),
        patch(f"{_PATH}.triggers_db", return_value=tdb),
    ):
        result = await UpdatePresetTool()._execute(
            user_id=_USER,
            session=scoped_session,
            preset_id="preset-a",
            name="Renamed",
        )

    assert isinstance(result, PresetUpdatedResponse)
    tdb.update_triggered_preset.assert_awaited_once()


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
    with (
        patch(f"{_PATH}.library_db", return_value=ldb),
        patch(f"{_PATH}.triggers_db", return_value=MagicMock()),
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
    with (
        patch(f"{_PATH}.library_db", return_value=ldb),
        patch(f"{_PATH}.triggers_db", return_value=tdb),
    ):
        result = await DeletePresetTool()._execute(
            user_id=_USER, session=session, preset_id="preset-1"
        )
    assert isinstance(result, PresetDeletedResponse)
    assert result.name == "ToDelete"
    tdb.delete_preset_with_webhook_cleanup.assert_awaited_once()


@pytest.mark.parametrize(
    ("session_expert_id", "target_expert_id"),
    [
        (None, "expert-a"),
        ("expert-a", None),
        ("expert-a", "expert-b"),
        ("expert-b", "expert-a"),
    ],
)
@pytest.mark.asyncio
async def test_delete_refuses_cross_persona_preset(session_expert_id, target_expert_id):
    scoped_session = make_session(_USER, expert_id=session_expert_id)
    ldb = MagicMock()
    ldb.get_preset = AsyncMock(return_value=_preset(expert_id=target_expert_id))
    tdb = MagicMock()
    tdb.delete_preset_with_webhook_cleanup = AsyncMock()

    with (
        patch(f"{_PATH}.library_db", return_value=ldb),
        patch(f"{_PATH}.triggers_db", return_value=tdb),
    ):
        result = await DeletePresetTool()._execute(
            user_id=_USER,
            session=scoped_session,
            preset_id="foreign-preset",
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "preset_not_found"
    tdb.delete_preset_with_webhook_cleanup.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_allows_same_expert_preset():
    scoped_session = make_session(_USER, expert_id="expert-a")
    ldb = MagicMock()
    ldb.get_preset = AsyncMock(
        return_value=_preset(name="Expert A preset", expert_id="expert-a")
    )
    tdb = MagicMock()
    tdb.delete_preset_with_webhook_cleanup = AsyncMock()

    with (
        patch(f"{_PATH}.library_db", return_value=ldb),
        patch(f"{_PATH}.triggers_db", return_value=tdb),
    ):
        result = await DeletePresetTool()._execute(
            user_id=_USER,
            session=scoped_session,
            preset_id="preset-a",
        )

    assert isinstance(result, PresetDeletedResponse)
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
    with (
        patch(f"{_PATH}.library_db", return_value=ldb),
        patch(f"{_PATH}.triggers_db", return_value=tdb),
    ):
        result = await UpdatePresetTool()._execute(
            user_id=_USER, session=session, preset_id="preset-1", inputs={"repo": "x"}
        )
    assert isinstance(result, ErrorResponse)
    assert result.error == "preset_update_failed"
    assert "provider refused" in result.message
