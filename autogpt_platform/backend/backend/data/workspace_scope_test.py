import pytest

from backend.data import workspace as workspace_module
from backend.data.workspace_scope import WorkspaceScope, resolve_expert_workspace_scope

SCOPE = WorkspaceScope(
    expert_id="expert-a",
    session_ids=["expert-a", "expert-a-old"],
    delegated_session_ids=["sub-1"],
)


@pytest.mark.parametrize(
    "path",
    ["/sessions/expert-a/file.txt", "/sessions/expert-a-old/nested/file.txt"],
)
def test_own_conversations_are_readable_and_writable(path: str):
    assert SCOPE.allows_path(path)
    assert SCOPE.allows_path(path, write=True)


@pytest.mark.parametrize(
    "path",
    [
        "/experts/expert-a/skills/mine/SKILL.md",
        "/experts/expert-a/skills/mine/references/notes.md",
    ],
)
def test_own_skills_folder_is_readable_and_writable(path: str):
    assert SCOPE.allows_path(path)
    assert SCOPE.allows_path(path, write=True)


def test_delegated_conversations_are_read_only():
    assert SCOPE.allows_path("/sessions/sub-1/result.json")
    assert not SCOPE.allows_path("/sessions/sub-1/result.json", write=True)


@pytest.mark.parametrize(
    "path",
    [
        "/sessions/expert-b/private.txt",
        "/sessions/personal/private.txt",
        "/sessions/expert-a-older/private.txt",
        "/sessions/expert-a/../expert-b/private.txt",
        "/sessions/expert-a//file.txt",
        "sessions/expert-a/file.txt",
        "/sessions/expert-a/..\\expert-b/x",
        "/skills",
        "/skills/autopilot-skill/SKILL.md",
        "/skills/autopilot-skill/references/notes.md",
        "/skillsets/private.txt",
        "/experts/expert-b/skills/theirs/SKILL.md",
        "/experts/expert-a/skillsets/mine.txt",
        "/root-file.txt",
    ],
)
def test_everything_else_is_denied(path: str):
    assert not SCOPE.allows_path(path)
    assert not SCOPE.allows_path(path, write=True)


def test_session_only_scope_carries_no_expert_grants():
    scope = WorkspaceScope(session_ids=["only"])
    assert scope.expert_id is None
    assert scope.skills_prefix is None
    assert scope.allows_path("/sessions/only/file.txt", write=True)
    assert not scope.allows_path("/sessions/other/file.txt")


def test_resolver_is_reachable_through_the_direct_db_accessor():
    assert (
        workspace_module.resolve_expert_workspace_scope
        is resolve_expert_workspace_scope
    )


def test_with_session_returns_new_scope_without_mutating():
    widened = SCOPE.with_session("expert-a-new")
    assert "expert-a-new" in widened.session_ids
    assert "expert-a-new" not in SCOPE.session_ids
    assert widened.with_session("expert-a-new") is widened
