"""Workspace paths and their SDK consumers, without launching a sandbox or CLI."""

import builtins
import importlib.util
import ntpath
import os
import posixpath
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.copilot import context, transcript
from backend.copilot.sdk import service
from backend.copilot.tools import sandbox


@pytest.fixture
def sandbox_on():
    def load(path_module: ModuleType, prefix: str | None = None):
        """Give only this module a different OS; pytest keeps its host paths."""
        local_os = SimpleNamespace(
            path=path_module,
            environ={} if prefix is None else {"COPILOT_WORKSPACE_PREFIX": prefix},
            makedirs=Mock(),
        )

        def import_for_sandbox(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "os":
                return local_os
            return builtins.__import__(name, globals, locals, fromlist, level)

        spec = importlib.util.spec_from_file_location(
            "sandbox_under_test", sandbox.__file__
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        module.__dict__["__builtins__"] = {
            **vars(builtins),
            "__import__": import_for_sandbox,
        }
        spec.loader.exec_module(module)
        return module

    return load


@pytest.mark.parametrize("path_module", [posixpath, ntpath], ids=["posix", "windows"])
@pytest.mark.parametrize("prefix", [None, ""])
def test_default_workspace_prefix(sandbox_on, path_module, prefix):
    module = sandbox_on(path_module, prefix)
    assert module.WORKSPACE_PREFIX == path_module.normpath("/tmp/copilot-")
    assert module.make_session_path("abc-123") == path_module.normpath(
        "/tmp/copilot-abc-123"
    )


@pytest.mark.parametrize(
    "path_module,prefix,expected",
    [
        (posixpath, "/srv/data/./tmp/copilot-", "/srv/data/tmp/copilot-abc-123"),
        (
            ntpath,
            "C:/Users/test/App Data/./tmp/copilot-",
            r"C:\Users\test\App Data\tmp\copilot-abc-123",
        ),
    ],
    ids=["posix", "windows"],
)
def test_relocated_workspace(sandbox_on, path_module, prefix, expected):
    module = sandbox_on(path_module, prefix)
    assert module.get_workspace_dir("abc-123") == expected
    module.os.makedirs.assert_called_once_with(expected, exist_ok=True)


@pytest.mark.parametrize("path_module", [posixpath, ntpath], ids=["posix", "windows"])
@pytest.mark.parametrize(
    "session_id,safe_id",
    [
        ("../../etc", "etc"),
        (r"..\..\Windows\system32", "Windowssystem32"),
        ("C:/temp/session:stream", "Ctempsessionstream"),
        ("abc_123\x00/名", "abc123"),
        ("", "default"),
        ("../\\", "default"),
    ],
)
def test_session_id_stays_within_workspace(
    sandbox_on, path_module, session_id, safe_id
):
    module = sandbox_on(path_module)
    path = module.make_session_path(session_id)
    assert path == path_module.normpath(f"/tmp/copilot-{safe_id}")
    assert path_module.dirname(path) == path_module.normpath("/tmp")


def test_sdk_imports_shared_workspace_prefix():
    assert service._SDK_CWD_PREFIX == sandbox.WORKSPACE_PREFIX
    assert service.make_session_path is sandbox.make_session_path


@pytest.mark.parametrize(
    "path_module,prefix",
    [
        (posixpath, None),
        (ntpath, None),
        (posixpath, "/srv/data/tmp/copilot-"),
        (ntpath, "C:/Users/test/App Data/tmp/copilot-"),
    ],
    ids=["posix-default", "windows-default", "posix-relocated", "windows-relocated"],
)
@pytest.mark.asyncio
async def test_sdk_cwd_and_cleanup(sandbox_on, monkeypatch, path_module, prefix):
    module = sandbox_on(path_module, prefix)
    monkeypatch.setattr(service, "make_session_path", module.make_session_path)
    monkeypatch.setattr(service, "_SDK_CWD_PREFIX", module.WORKSPACE_PREFIX)
    monkeypatch.setattr(service, "os", module.os)
    monkeypatch.setattr(context, "os", module.os)
    monkeypatch.setattr(service, "_last_sweep_time", 0.0)
    remove = Mock()
    sweep = Mock()
    monkeypatch.setattr(service, "shutil", SimpleNamespace(rmtree=remove))
    monkeypatch.setattr(service, "cleanup_stale_project_dirs", sweep)

    cwd = service._make_sdk_cwd("../../abc-123")
    expected = path_module.normpath(f"{prefix or '/tmp/copilot-'}abc-123")
    assert cwd == expected
    await service._cleanup_sdk_tool_results(cwd)

    remove.assert_called_once_with(expected, True)
    sweep.assert_called_once_with(context.encode_cwd_for_cli(expected))
    assert "-tmp-copilot-" in sweep.call_args.args[0]


@pytest.mark.parametrize("path_module", [posixpath, ntpath], ids=["posix", "windows"])
@pytest.mark.asyncio
async def test_sdk_rejects_paths_outside_workspace(
    sandbox_on, monkeypatch, path_module
):
    module = sandbox_on(path_module)
    monkeypatch.setattr(service, "_SDK_CWD_PREFIX", module.WORKSPACE_PREFIX)
    monkeypatch.setattr(service, "os", module.os)
    remove = Mock()
    sweep = Mock()
    monkeypatch.setattr(service, "shutil", SimpleNamespace(rmtree=remove))
    monkeypatch.setattr(service, "cleanup_stale_project_dirs", sweep)

    for cwd in ("/tmp/other-session", "/tmp/copilot-abc/../../outside"):
        monkeypatch.setattr(service, "make_session_path", Mock(return_value=cwd))
        with pytest.raises(ValueError, match="SDK cwd escaped prefix"):
            service._make_sdk_cwd("abc")
        await service._cleanup_sdk_tool_results(cwd)

    remove.assert_not_called()
    sweep.assert_not_called()


@pytest.mark.parametrize("workspace_dir,removed", [("tmp", 1), ("workspaces", 0)])
def test_relocated_transcript_cleanup_stays_scoped(
    sandbox_on, monkeypatch, tmp_path: Path, workspace_dir, removed
):
    module = sandbox_on(os.path, str(tmp_path / "data" / workspace_dir / "copilot-"))
    projects = tmp_path / "projects"
    projects.mkdir()
    monkeypatch.setattr(transcript, "projects_base", lambda: str(projects))
    target_name = context.encode_cwd_for_cli(module.make_session_path("target"))
    other_name = context.encode_cwd_for_cli(module.make_session_path("other"))
    old_time = time.time() - transcript._STALE_PROJECT_DIR_SECONDS - 100
    for name in (target_name, other_name):
        directory = projects / name
        directory.mkdir()
        (directory / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
        os.utime(directory, (old_time, old_time))

    assert transcript.cleanup_stale_project_dirs(target_name) == removed
    assert (projects / target_name).exists() == (removed == 0)
    assert (projects / other_name / "transcript.jsonl").is_file()
