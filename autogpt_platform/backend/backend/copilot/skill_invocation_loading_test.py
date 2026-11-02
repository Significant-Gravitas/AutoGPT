import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.copilot import skill_invocation as si
from backend.copilot.tools import skills
from backend.data.skill_versions import content_hash


@pytest.fixture
def command_load(mocker):
    parsed = skills.ParsedSkill(
        name="report", description="report", body="Write $ARGUMENTS"
    )
    text = skills.render_skill_markdown(parsed)
    manager = mocker.Mock()
    manager.read_file = AsyncMock(return_value=text.encode())
    mocker.patch.object(si, "resolve_skill_scope", AsyncMock(return_value=None))
    mocker.patch.object(si, "read_user_skill_with_body", AsyncMock(return_value=parsed))
    mocker.patch.object(si, "is_skills_feature_enabled", AsyncMock(return_value=True))
    mocker.patch.object(
        si, "persist_current_user_message", AsyncMock(side_effect=lambda *a, **kw: a[2])
    )
    mocker.patch.object(
        skills, "_get_user_skill_manager", AsyncMock(return_value=manager)
    )
    reconcile = mocker.patch(
        "backend.copilot.learning.publish.reconcile_skill_for_read",
        new_callable=AsyncMock,
    )
    head = SimpleNamespace(use_paused_at=None, current_version_id="v1")
    current = SimpleNamespace(
        state="ready",
        content_hash=content_hash(text),
        files=None,
        version=1,
        id="v1",
        origin="imported",
    )
    mocker.patch.object(
        skills, "resolve_current_version", AsyncMock(return_value=(head, current))
    )
    loads = mocker.patch.object(skills, "record_skill_loaded", new_callable=AsyncMock)
    return head, current, loads, reconcile


@pytest.mark.parametrize("state", ["paused", "pending"])
async def test_command_refuses_unavailable_skill(command_load, state, caplog):
    head, current, loads, _ = command_load
    if state == "paused":
        head.use_paused_at = "paused"
    else:
        current.state = "pending"
    assert (
        await si.inject_skill_invocation(
            "/report weekly", "session", [], user_id="user", expert_id=None
        )
        is None
    )
    loads.assert_not_awaited()
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]


async def test_command_records_the_reconciled_loaded_version(command_load):
    _, _, loads, reconcile = command_load
    message = await si.inject_skill_invocation(
        "/report weekly", "session", [], user_id="user", expert_id=None
    )
    assert message is not None and "Write weekly" in message
    reconcile.assert_awaited_once_with("user", None, "report")
    loads.assert_awaited_once_with(
        "user", None, "report", version_id="v1", session_id="session"
    )


async def test_command_logs_unexpected_load_errors(command_load, mocker, caplog):
    mocker.patch.object(
        skills,
        "resolve_loaded_skill_version",
        AsyncMock(side_effect=RuntimeError("registry unavailable")),
    )
    assert (
        await si.inject_skill_invocation(
            "/report weekly", "session", [], user_id="user", expert_id=None
        )
        is None
    )
    assert any(
        record.levelno >= logging.ERROR and record.exc_info for record in caplog.records
    )
