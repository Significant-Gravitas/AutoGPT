"""Tests for context.py — execution context variables and path helpers."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from backend.copilot.context import (
    SDK_PROJECTS_DIR,
    _current_project_dir,
    get_current_permissions,
    get_current_sandbox,
    get_execution_context,
    get_sdk_cwd,
    is_allowed_local_path,
    resolve_sandbox_path,
    set_execution_context,
)
from backend.copilot.permissions import CopilotPermissions


def _make_session() -> MagicMock:
    s = MagicMock()
    s.session_id = "test-session"
    return s


# ---------------------------------------------------------------------------
# Context variable getters
# ---------------------------------------------------------------------------


def test_get_execution_context_defaults():
    """get_execution_context returns (None, session) when user_id is not set."""
    set_execution_context(None, _make_session())
    user_id, session = get_execution_context()
    assert user_id is None
    assert session is not None


def test_set_and_get_execution_context():
    """set_execution_context stores user_id and session."""
    mock_session = _make_session()
    set_execution_context("user-abc", mock_session)
    user_id, session = get_execution_context()
    assert user_id == "user-abc"
    assert session is mock_session


def test_get_current_sandbox_none_by_default():
    """get_current_sandbox returns None when no sandbox is set."""
    set_execution_context("u1", _make_session(), sandbox=None)
    assert get_current_sandbox() is None


def test_get_current_sandbox_returns_set_value():
    """get_current_sandbox returns the sandbox set via set_execution_context."""
    mock_sandbox = MagicMock()
    set_execution_context("u1", _make_session(), sandbox=mock_sandbox)
    assert get_current_sandbox() is mock_sandbox


def test_set_and_get_current_permissions():
    """set_execution_context stores permissions; get_current_permissions returns it."""
    perms = CopilotPermissions(tools=["run_block"], tools_exclude=False)
    set_execution_context("u1", _make_session(), permissions=perms)
    assert get_current_permissions() is perms


def test_get_current_permissions_defaults_to_none():
    """get_current_permissions returns None when no permissions have been set."""
    set_execution_context("u1", _make_session())
    assert get_current_permissions() is None


def test_get_sdk_cwd_empty_when_not_set():
    """get_sdk_cwd returns empty string when sdk_cwd is not set."""
    set_execution_context("u1", _make_session(), sdk_cwd=None)
    assert get_sdk_cwd() == ""


def test_get_sdk_cwd_returns_set_value():
    """get_sdk_cwd returns the value set via set_execution_context."""
    set_execution_context("u1", _make_session(), sdk_cwd="/tmp/copilot-test")
    assert get_sdk_cwd() == "/tmp/copilot-test"


# ---------------------------------------------------------------------------
# is_allowed_local_path
# ---------------------------------------------------------------------------


def test_is_allowed_local_path_empty():
    assert not is_allowed_local_path("")


def test_is_allowed_local_path_inside_sdk_cwd():
    with tempfile.TemporaryDirectory() as cwd:
        path = os.path.join(cwd, "file.txt")
        assert is_allowed_local_path(path, cwd)


def test_is_allowed_local_path_sdk_cwd_itself():
    with tempfile.TemporaryDirectory() as cwd:
        assert is_allowed_local_path(cwd, cwd)


def test_is_allowed_local_path_outside_sdk_cwd():
    with tempfile.TemporaryDirectory() as cwd:
        assert not is_allowed_local_path("/etc/passwd", cwd)


def test_is_allowed_local_path_no_sdk_cwd_no_project_dir():
    """Without sdk_cwd or project_dir, all paths are rejected."""
    _current_project_dir.set("")
    assert not is_allowed_local_path("/tmp/some-file.txt", sdk_cwd=None)


def test_is_allowed_local_path_tool_results_with_uuid():
    """Files under <encoded-cwd>/<uuid>/tool-results/ are allowed."""
    encoded = "test-encoded-dir"
    conv_uuid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    path = os.path.join(
        SDK_PROJECTS_DIR, encoded, conv_uuid, "tool-results", "output.txt"
    )

    _current_project_dir.set(encoded)
    try:
        assert is_allowed_local_path(path, sdk_cwd=None)
    finally:
        _current_project_dir.set("")


def test_is_allowed_local_path_tool_outputs_with_uuid():
    """Files under <encoded-cwd>/<uuid>/tool-outputs/ are also allowed."""
    encoded = "test-encoded-dir"
    conv_uuid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    path = os.path.join(
        SDK_PROJECTS_DIR, encoded, conv_uuid, "tool-outputs", "output.json"
    )

    _current_project_dir.set(encoded)
    try:
        assert is_allowed_local_path(path, sdk_cwd=None)
    finally:
        _current_project_dir.set("")


def test_is_allowed_local_path_tool_results_without_uuid_rejected():
    """Direct <encoded-cwd>/tool-results/ (no UUID) is rejected."""
    encoded = "test-encoded-dir"
    path = os.path.join(SDK_PROJECTS_DIR, encoded, "tool-results", "output.txt")

    _current_project_dir.set(encoded)
    try:
        assert not is_allowed_local_path(path, sdk_cwd=None)
    finally:
        _current_project_dir.set("")


def test_is_allowed_local_path_sibling_of_tool_results_is_rejected():
    """A path adjacent to tool-results/ but not inside it is rejected."""
    encoded = "test-encoded-dir"
    sibling_path = os.path.join(SDK_PROJECTS_DIR, encoded, "other-dir", "file.txt")

    _current_project_dir.set(encoded)
    try:
        assert not is_allowed_local_path(sibling_path, sdk_cwd=None)
    finally:
        _current_project_dir.set("")


def test_is_allowed_local_path_valid_uuid_wrong_segment_name_rejected():
    """A valid UUID dir but non-'tool-results'/'tool-outputs' second segment is rejected."""
    encoded = "test-encoded-dir"
    uuid_str = "12345678-1234-5678-9abc-def012345678"
    path = os.path.join(
        SDK_PROJECTS_DIR, encoded, uuid_str, "not-tool-results", "output.txt"
    )

    _current_project_dir.set(encoded)
    try:
        assert not is_allowed_local_path(path, sdk_cwd=None)
    finally:
        _current_project_dir.set("")


# ---------------------------------------------------------------------------
# resolve_sandbox_path
# ---------------------------------------------------------------------------


def test_resolve_sandbox_path_absolute_valid():
    assert (
        resolve_sandbox_path("/home/user/project/main.py")
        == "/home/user/project/main.py"
    )


def test_resolve_sandbox_path_relative():
    assert resolve_sandbox_path("project/main.py") == "/home/user/project/main.py"


def test_resolve_sandbox_path_workdir_itself():
    assert resolve_sandbox_path("/home/user") == "/home/user"


def test_resolve_sandbox_path_normalizes_dots():
    assert resolve_sandbox_path("/home/user/a/../b") == "/home/user/b"


def test_resolve_sandbox_path_escape_raises():
    with pytest.raises(ValueError, match="must be within"):
        resolve_sandbox_path("/home/user/../../etc/passwd")


def test_resolve_sandbox_path_absolute_outside_raises():
    with pytest.raises(ValueError):
        resolve_sandbox_path("/etc/passwd")


def test_resolve_sandbox_path_tmp_allowed():
    assert resolve_sandbox_path("/tmp/data.txt") == "/tmp/data.txt"


def test_resolve_sandbox_path_tmp_nested():
    assert resolve_sandbox_path("/tmp/a/b/c.txt") == "/tmp/a/b/c.txt"


def test_resolve_sandbox_path_tmp_itself():
    assert resolve_sandbox_path("/tmp") == "/tmp"


def test_resolve_sandbox_path_tmp_escape_raises():
    with pytest.raises(ValueError):
        resolve_sandbox_path("/tmp/../etc/passwd")


def test_resolve_sandbox_path_tmp_prefix_collision_raises():
    with pytest.raises(ValueError):
        resolve_sandbox_path("/tmp_evil/malicious.txt")
