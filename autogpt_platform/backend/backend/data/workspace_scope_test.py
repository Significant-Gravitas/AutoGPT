import pytest

from backend.data import workspace as workspace_module
from backend.data.workspace_scope import WorkspaceScope, resolve_expert_workspace_scope

SCOPE = WorkspaceScope(
    expert_id="expert-a",
    owns_skills_folder=True,
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


def test_denial_survives_the_rpc_round_trip():
    """The scope is rebuilt as this class on the client side, so a denial that
    lived in a subclass would come back as a grant."""
    denied = WorkspaceScope(expert_id="expert-a")
    rebuilt = WorkspaceScope.model_validate(denied.model_dump())

    for scope in (denied, rebuilt):
        assert scope.skills_prefix is None
        assert not scope.allows_path("/experts/expert-a/skills/x/SKILL.md")
        assert not scope.allows_path("/experts/expert-a/skills/x/SKILL.md", write=True)
    assert WorkspaceScope.model_validate(SCOPE.model_dump()).allows_path(
        "/experts/expert-a/skills/mine/SKILL.md", write=True
    )


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


# Same grants plus the owner's own files, which is what
# ``resolve_expert_workspace_scope`` builds for a live hired expert.
USER_FILES_SCOPE = SCOPE.model_copy(update={"reads_user_files": True})


@pytest.mark.parametrize(
    "path",
    [
        "/root-file.txt",
        "/Invoices/2026/march.pdf",
        "/sessionsish/file.txt",
        "/skillsets/private.txt",
    ],
)
def test_user_files_are_readable_but_never_writable(path: str):
    assert USER_FILES_SCOPE.allows_path(path)
    assert not USER_FILES_SCOPE.allows_path(path, write=True)


@pytest.mark.parametrize(
    "path",
    [
        "/sessions/expert-b/private.txt",
        "/sessions/personal/private.txt",
        "/skills/autopilot-skill/SKILL.md",
        "/experts/expert-b/skills/theirs/SKILL.md",
        "/sessions/expert-a/../../root-file.txt",
        "root-file.txt",
    ],
)
def test_the_user_files_grant_opens_nothing_under_the_managed_roots(path: str):
    assert not USER_FILES_SCOPE.allows_path(path)


def test_the_user_files_grant_survives_the_rpc_round_trip():
    """Carried as a field, so a scope that denies user files cannot come back
    granting them, and one that grants them cannot come back denying them."""
    for scope in (SCOPE, USER_FILES_SCOPE):
        rebuilt = WorkspaceScope.model_validate(scope.model_dump())
        assert rebuilt.reads_user_files == scope.reads_user_files
        assert rebuilt.allows_path("/root-file.txt") == scope.allows_path(
            "/root-file.txt"
        )
