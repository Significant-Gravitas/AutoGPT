import hashlib
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.copilot import skill_invocation as si
from backend.copilot.tools import skills
from backend.data.skill_version_files import SkillVersionFile
from backend.data.skill_versions import content_hash

_WORKDIR = "/home/user"


@pytest.fixture
def command_load(mocker):
    parsed = skills.ParsedSkill(
        name="report", description="report", body="Write $ARGUMENTS"
    )
    text = skills.render_skill_markdown(parsed)
    manager = mocker.Mock()
    manager.read_file = AsyncMock(return_value=text.encode())
    mocker.patch.object(si, "resolve_skill_scope", AsyncMock(return_value=None))
    mocker.patch.object(si, "list_user_skills", AsyncMock(return_value=[parsed]))
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
    # The working directory the package is copied into, as the turn's shell
    # sees it; nothing here touches the disk.
    mocker.patch.object(skills, "_list_package_files", AsyncMock(return_value=[]))
    mocker.patch.object(skills, "workdir_root", return_value=_WORKDIR)
    mocker.patch.object(skills, "_read_package_manifest", AsyncMock(return_value={}))
    mocker.patch.object(skills, "_write_package_manifest", AsyncMock())
    mocker.patch.object(skills, "set_executable", AsyncMock(return_value=[]))
    mocker.patch.object(skills, "remove_from_workdir", AsyncMock(return_value=[]))
    mocker.patch.object(
        skills, "save_to_workdir", AsyncMock(side_effect=lambda path, *_: path)
    )
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


@pytest.mark.parametrize("command", ["report", "team:report"])
async def test_selected_skill_is_read_once_after_reconcile(
    command_load, mocker, command
):
    _, _, loads, reconcile = command_load
    events = []
    manager = skills._get_user_skill_manager.return_value
    content = manager.read_file.return_value

    async def read(path):
        events.append("read")
        return content

    manager.read_file.side_effect = read
    reconcile.side_effect = lambda *args: events.append("reconcile")
    loads.side_effect = lambda *args, **kwargs: events.append("record")
    original_resolve = skills.resolve_loaded_skill_version

    async def resolve(*args, **kwargs):
        events.append("resolve")
        return await original_resolve(*args, **kwargs)

    mocker.patch.object(skills, "resolve_loaded_skill_version", side_effect=resolve)
    mocker.patch.object(
        si,
        "list_user_skills",
        AsyncMock(
            return_value=[
                skills.ParsedSkill(
                    name="report",
                    description="Report",
                    body="",
                    aliases=("team:report",),
                )
            ]
        ),
    )
    result = await si.resolve_invocable_skill("user", None, command, "session")
    assert result is not None and result.skill.body.strip() == "Write $ARGUMENTS"
    assert events == ["reconcile", "read", "resolve", "record"]
    manager.read_file.assert_awaited_once_with("/skills/report/SKILL.md")


async def test_command_points_the_skill_dir_where_its_package_is_copied(
    command_load, mocker
):
    """``${CLAUDE_SKILL_DIR}`` names the copy in the shell's working
    directory, where read_skill puts the package too, not the workspace
    folder the package is stored in."""
    _, current, loads, _ = command_load
    body = "Run python ${CLAUDE_SKILL_DIR}/scripts/run.py $ARGUMENTS"
    text = skills.render_skill_markdown(
        skills.ParsedSkill(name="report", description="report", body=body)
    )
    current.content_hash = content_hash(text)
    script = skills.SkillFileInfo(
        path="/skills/report/scripts/run.py", file_id="f1", is_executable=True
    )
    manager = skills._get_user_skill_manager.return_value
    manager.read_file = AsyncMock(
        side_effect=lambda path: {
            "/skills/report/SKILL.md": text.encode(),
            script.path: b"print('report')",
        }[path]
    )
    mocker.patch.object(skills, "_list_package_files", AsyncMock(return_value=[script]))

    message = await si.inject_skill_invocation(
        "/report weekly", "session", [], user_id="user", expert_id=None
    )

    copied_to = f"{_WORKDIR}/skills/report/scripts/run.py"
    assert message is not None
    assert f"Run python {copied_to} weekly" in message
    assert "Run python /skills/report/" not in message
    assert f"Package files are at {_WORKDIR}/skills/report;" in message
    written = [call.args[0] for call in skills.save_to_workdir.await_args_list]
    assert written == [copied_to]
    loads.assert_awaited_once()


@pytest.mark.parametrize("second_listing", ["changed", "unavailable"])
async def test_command_is_not_expanded_when_its_package_cannot_go_with_it(
    command_load, mocker, caplog, second_listing
):
    """A package that changed or cannot be listed between the version check
    and the copy is refused, as read_skill refuses it; the command then goes
    to the model as typed."""
    _, current, loads, _ = command_load
    script = skills.SkillFileInfo(
        path="/skills/report/scripts/run.py",
        file_id="f1",
        checksum=hashlib.sha256(b"print()").hexdigest(),
    )
    current.files = [
        SkillVersionFile(relative_path="scripts/run.py", content_base64="cHJpbnQoKQ==")
    ]
    later = [] if second_listing == "changed" else RuntimeError("storage down")
    mocker.patch.object(
        skills, "_list_package_files", AsyncMock(side_effect=[[script], later])
    )
    assert (
        await si.inject_skill_invocation(
            "/report weekly", "session", [], user_id="user", expert_id=None
        )
        is None
    )
    loads.assert_not_awaited()
    skills.save_to_workdir.assert_not_awaited()
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
