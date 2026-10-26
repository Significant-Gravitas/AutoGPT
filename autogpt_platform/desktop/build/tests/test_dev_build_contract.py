"""Current-dev dependency and build boundaries, without assembling a runtime."""

import importlib
import sys
from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUILD))
lock_export = importlib.import_module("lock_export")
build_runtime = importlib.import_module("build_runtime")
LOCK = BUILD.parents[1] / "backend" / "poetry.lock"


def test_current_dev_moviepy_uses_the_locked_git_commit():
    lines = lock_export.export(LOCK)
    assert (
        "moviepy @ git+https://github.com/Zulko/moviepy.git@"
        "4cf7f84ed1f840e4f0c3ed92b8cdec12a87b56ec"
    ) in lines
    assert "moviepy" not in lock_export.excluded(LOCK)


@pytest.mark.parametrize("revision", ["main", "v2.1.2", "abcd123", ""])
def test_git_dependencies_require_a_resolved_full_commit(revision):
    package = {
        "name": "moviepy",
        "version": "2.1.2",
        "source": {
            "type": "git",
            "url": "https://github.com/Zulko/moviepy.git",
            "reference": "main",
            "resolved_reference": revision,
        },
    }
    with pytest.raises(RuntimeError, match="immutable"):
        lock_export.requirement(package)


@pytest.mark.parametrize("kind", ["directory", "file", "url", "legacy"])
def test_unhandled_dependency_sources_stop_the_build(kind):
    package = {
        "name": "new-dependency",
        "version": "1.0",
        "source": {"type": kind, "url": "../other"},
    }
    with pytest.raises(RuntimeError, match="unsupported source"):
        lock_export.requirement(package)


def test_refresh_synchronizes_and_copies_all_dependencies(monkeypatch, tmp_path):
    build = build_runtime.Build(tmp_path / "runtime", tmp_path / "cache")
    build.cache.mkdir()
    commands = []
    monkeypatch.setattr(
        build_runtime, "run", lambda command, **kwargs: commands.append(command)
    )
    monkeypatch.setattr(
        build_runtime.claude_cli,
        "ensure",
        lambda *args: build.site_packages / "claude_agent_sdk/_bundled/claude",
    )
    monkeypatch.setattr(
        build_runtime.claude_cli, "declared_version", lambda *args: "test"
    )
    build.step_deps()
    sync = commands[0]
    assert sync[:3] == ["uv", "pip", "sync"]
    assert "--reinstall" in sync
    assert sync[sync.index("--link-mode") + 1] == "copy"
    requirements = (build.cache / "requirements.txt").read_text("utf-8")
    assert "autogpt-libs @ file:" in requirements
    assert commands[-1][:3] == ["uv", "pip", "check"]


def test_frontend_build_respects_explicit_heap_limit(monkeypatch, tmp_path):
    build = build_runtime.Build(tmp_path / "runtime", tmp_path / "cache")
    monkeypatch.setenv("AUTOGPT_DESKTOP_BUILD_HEAP_MB", "4096")
    commands = []
    monkeypatch.setattr(build, "_pnpm", lambda env: ["pnpm"])
    monkeypatch.setattr(
        build_runtime,
        "run",
        lambda command, **kwargs: commands.append((command, kwargs)),
    )
    monkeypatch.setattr(build_runtime.shutil, "copytree", lambda *args, **kwargs: None)
    monkeypatch.setattr(build_runtime.shutil, "rmtree", lambda *args, **kwargs: None)
    monkeypatch.setattr(build, "step_frontend_config", lambda: None)
    build.step_frontend()
    assert "--prod=false" in commands[0][0]
    assert commands[-2][0] == ["pnpm", "run", "copy:vad-assets"]
    assert commands[-1][0] == ["pnpm", "exec", "next", "build"]
    assert commands[-1][1]["env"]["NODE_OPTIONS"] == "--max-old-space-size=4096"
