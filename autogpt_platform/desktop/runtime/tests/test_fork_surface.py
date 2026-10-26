"""Keep the desktop fork's shared-source changes explicitly bounded.

The installer and runtime live under autogpt_platform/desktop. The caller
workflow and the exact shared files needed for supported features and reliable
chat display are listed below. Other shared-source differences remain forbidden.

The rule is the `desktop` branch's. A variant (`variant/<slug>`, README
"Variants") exists to differ from upstream, and is not held to it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from autogpt_desktop import settings
from autogpt_desktop.layout import Bundle, DataDir

REPO = Path(__file__).resolve().parents[4]
DESKTOP_BRANCH = "desktop"
VARIANT_PREFIX = "variant/"
# What the build on `main` tells the unit tests (desktop-build.yml): the slug
# of the variant being built, empty for the normal app.
VARIANT_VARIABLE = "AUTOGPT_DESKTOP_VARIANT"
UPSTREAM = re.compile(
    r"github\.com[:/]+Significant-Gravitas/AutoGPT(?:\.git)?/?$", re.I
)
UPSTREAM_BRANCH = "dev"
OURS = "autogpt_platform/desktop/"
ALLOWED = {
    ".secrets.baseline",
    ".github/workflows/platform-desktop-build.yml",
    "autogpt_platform/backend/backend/api/features/integrations/codex.py",
    "autogpt_platform/backend/backend/api/features/integrations/incremental_oauth_test.py",
    "autogpt_platform/backend/backend/api/features/integrations/router.py",
    "autogpt_platform/backend/backend/blocks/linear/_oauth.py",
    "autogpt_platform/backend/backend/copilot/executor/processor.py",
    "autogpt_platform/backend/backend/copilot/executor/processor_cancel_test.py",
    "autogpt_platform/backend/backend/copilot/executor/processor_test.py",
    "autogpt_platform/backend/backend/copilot/executor/processor_budget_gate_test.py",
    "autogpt_platform/backend/backend/copilot/executor/scheduled_turn_alert_test.py",
    "autogpt_platform/backend/backend/copilot/tools/bash_exec.py",
    "autogpt_platform/backend/backend/copilot/tools/bash_exec_availability_test.py",
    "autogpt_platform/backend/backend/copilot/stream_registry.py",
    "autogpt_platform/backend/backend/copilot/stream_registry_test.py",
    "autogpt_platform/backend/backend/copilot/stream_registry_terminal_failure_test.py",
    "autogpt_platform/backend/backend/copilot/stream_registry_tracing_test.py",
    "autogpt_platform/backend/backend/copilot/tools/agent_browser.py",
    "autogpt_platform/backend/backend/copilot/tools/agent_browser_desktop_test.py",
    "autogpt_platform/backend/backend/copilot/sdk/tool_adapter.py",
    "autogpt_platform/backend/backend/copilot/sdk/tool_adapter_test.py",
    "autogpt_platform/backend/backend/copilot/sdk/subscription.py",
    "autogpt_platform/backend/backend/copilot/sdk/subscription_test.py",
    "autogpt_platform/backend/backend/copilot/sdk/env.py",
    "autogpt_platform/backend/backend/copilot/sdk/env_test.py",
    "autogpt_platform/backend/backend/data/redis_helpers.py",
    "autogpt_platform/backend/backend/data/redis_helpers_test.py",
    "autogpt_platform/backend/backend/data/db_manager.py",
    "autogpt_platform/backend/backend/data/user.py",
    "autogpt_platform/backend/backend/data/user_credential_refresh_test.py",
    "autogpt_platform/backend/backend/integrations/codex/chatgpt_auth.py",
    "autogpt_platform/backend/backend/integrations/codex/chatgpt_auth_test.py",
    "autogpt_platform/backend/backend/integrations/codex/transport.py",
    "autogpt_platform/backend/backend/integrations/codex/transport_test.py",
    "autogpt_platform/backend/backend/integrations/credential_refresh.py",
    "autogpt_platform/backend/backend/integrations/credentials_store.py",
    "autogpt_platform/backend/backend/integrations/credentials_store_upsert_test.py",
    "autogpt_platform/backend/backend/integrations/creds_manager.py",
    "autogpt_platform/backend/backend/integrations/creds_manager_test.py",
    "autogpt_platform/backend/backend/integrations/creds_manager_lock_test.py",
    "autogpt_platform/backend/backend/integrations/creds_manager_refresh_test.py",
    "autogpt_platform/frontend/src/app/(platform)/admin/memory/page.tsx",
    "autogpt_platform/frontend/src/app/(platform)/admin/memory/components/MemoryInspector.tsx",
    "autogpt_platform/frontend/src/app/(platform)/admin/memory/__tests__/memory-inspector.test.tsx",
    "autogpt_platform/frontend/src/app/(platform)/copilot/helpers/stripReplayPrefix.ts",
    "autogpt_platform/frontend/src/app/(platform)/copilot/helpers/__tests__/stripReplayPrefix.test.ts",
    "autogpt_platform/frontend/src/app/(platform)/settings/memory/page.tsx",
    "autogpt_platform/frontend/src/app/(platform)/settings/memory/__tests__/understanding.test.tsx",
    "autogpt_platform/frontend/src/components/layout/Navbar/components/AccountMenu/__tests__/AccountMenu.test.tsx",
    "autogpt_platform/frontend/src/services/environment/index.ts",
}
RUNTIME_CONFIG = "autogpt_platform/single-container/runtime_config.py"
PATCH = f"{OURS}upstream/runtime-config-windows.patch"


# --- which branch is this? -------------------------------------------------------


def rule_applies(
    environment: Mapping[str, str], branch: str | None, in_desktop: bool
) -> tuple[bool, str]:
    """Whether this checkout is held to the rule, and why.

    CI says which branch a commit belongs to: the target of a pull request
    (GITHUB_BASE_REF), else the branch that was pushed (GITHUB_REF_NAME).
    Without either it is git's to say: `branch` is the one checked out (None
    for a detached HEAD) and `in_desktop` whether HEAD belongs to `desktop`
    (head_is_in_desktop)."""
    if variant := environment.get(VARIANT_VARIABLE):
        return False, f"this is the variant {variant!r} ({VARIANT_VARIABLE})"
    for variable in ("GITHUB_BASE_REF", "GITHUB_REF_NAME"):
        named = environment.get(variable)
        if named == DESKTOP_BRANCH:
            return True, f"{variable} is {DESKTOP_BRANCH}"
        if named and named.startswith(VARIANT_PREFIX):
            return False, f"{variable} is the variant branch {named}"
        if named:
            break  # some other branch (the build is started from `main`): ask git
    if branch == DESKTOP_BRANCH:
        return True, f"{DESKTOP_BRANCH} is checked out"
    if branch and branch.startswith(VARIANT_PREFIX):
        return False, f"the variant branch {branch} is checked out"
    if in_desktop:
        return True, f"HEAD is part of {DESKTOP_BRANCH}"
    where = f"the branch {branch}" if branch else "a detached HEAD"
    return (
        False,
        f"{where} is not part of {DESKTOP_BRANCH}, the only branch the rule is for",
    )


def foreign(paths: list[str]) -> list[str]:
    """The paths that are neither the desktop's nor allowed."""
    return sorted(
        {
            path
            for path in paths
            if path and not path.startswith(OURS) and path not in ALLOWED
        }
    )


@pytest.mark.parametrize(
    ("environment", "branch", "in_desktop", "applies"),
    [
        # CI: a push to desktop, a pull request into it, and the same for a variant.
        ({"GITHUB_REF_NAME": "desktop"}, None, False, True),
        (
            {"GITHUB_BASE_REF": "desktop", "GITHUB_REF_NAME": "12/merge"},
            None,
            False,
            True,
        ),
        ({"GITHUB_REF_NAME": "variant/voice"}, None, True, False),
        (
            {"GITHUB_BASE_REF": "variant/voice", "GITHUB_REF_NAME": "7/merge"},
            None,
            True,
            False,
        ),
        # A pull request from a variant into desktop is held to desktop's rule.
        (
            {"GITHUB_BASE_REF": "desktop", "GITHUB_REF_NAME": "variant/voice"},
            None,
            False,
            True,
        ),
        # GitHub sets GITHUB_BASE_REF to "" outside pull requests.
        ({"GITHUB_BASE_REF": "", "GITHUB_REF_NAME": "desktop"}, None, False, True),
        # The build on main says which variant it builds, whatever started it.
        (
            {"AUTOGPT_DESKTOP_VARIANT": "voice", "GITHUB_REF_NAME": "desktop"},
            "desktop",
            True,
            False,
        ),
        (
            {"AUTOGPT_DESKTOP_VARIANT": "", "GITHUB_REF_NAME": "desktop"},
            None,
            False,
            True,
        ),
        # Started from main (the sync, a manual build): git decides.
        ({"GITHUB_REF_NAME": "main"}, "desktop", True, True),
        ({"GITHUB_REF_NAME": "main"}, None, True, True),
        ({"GITHUB_REF_NAME": "main"}, None, False, False),
        ({"GITHUB_REF_NAME": "main"}, "variant/voice", False, False),
        # A developer's checkout.
        ({}, "desktop", True, True),
        ({}, "variant/voice", False, False),
        ({}, "variant/voice", True, False),
        ({}, None, True, True),
        ({}, None, False, False),
        ({}, "some-work", True, True),
        ({}, "some-work", False, False),
    ],
)
def test_the_rule_is_for_the_desktop_branch_only(
    environment: dict[str, str], branch: str | None, in_desktop: bool, applies: bool
):
    decided, reason = rule_applies(environment, branch, in_desktop)
    assert decided is applies, reason
    assert reason


def test_only_the_desktop_directory_and_named_readiness_files_are_ours():
    assert foreign(
        [
            "autogpt_platform/desktop/README.md",
            ".github/workflows/platform-desktop-build.yml",
            *ALLOWED,
            "autogpt_platform/backend/backend/copilot/tools/bash_exec.py.bak",
            "autogpt_platform/backend/backend/integrations/credential_refresh.py.bak",
            "autogpt_platform/backend/backend/integrations/unreviewed_oauth.py",
            "autogpt_platform/frontend/src/components/molecules/TallyPoup/useTallyPopup.ts.bak",
            "autogpt_platform/frontend/src/components/molecules/TallyPoup/unreviewed.tsx",
            ".github/workflows/platform-desktop-build.yml.bak",
            "autogpt_platform/desktop.md",
            "autogpt_platform/single-container/runtime_config.py",
            ".gitignore",
            ".gitignore",
            "",
        ]
    ) == [
        ".github/workflows/platform-desktop-build.yml.bak",
        ".gitignore",
        "autogpt_platform/backend/backend/copilot/tools/bash_exec.py.bak",
        "autogpt_platform/backend/backend/integrations/credential_refresh.py.bak",
        "autogpt_platform/backend/backend/integrations/unreviewed_oauth.py",
        "autogpt_platform/desktop.md",
        "autogpt_platform/frontend/src/components/molecules/TallyPoup/unreviewed.tsx",
        "autogpt_platform/frontend/src/components/molecules/TallyPoup/useTallyPopup.ts.bak",
        "autogpt_platform/single-container/runtime_config.py",
    ]


# --- the rule, on this checkout ---------------------------------------------------


def run_git(*arguments: str) -> subprocess.CompletedProcess[str] | None:
    """None when there is no git to run."""
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def git(*arguments: str) -> str | None:
    """The command's output, or None when it failed or there is no git."""
    result = run_git(*arguments)
    return result.stdout.strip() if result and result.returncode == 0 else None


def is_ancestor(older: str, newer: str) -> bool:
    return git("merge-base", "--is-ancestor", older, newer) is not None


def head_is_in_desktop(branch: str | None) -> bool:
    """Whether HEAD is the tip of, or behind, a ref named `desktop`; or, for
    a branch with a name of its own, whether it was started from `desktop`
    and not from a variant: the one of the two HEAD has added the fewest
    commits to."""
    refs = (
        git("for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes") or ""
    ).split()
    desktops = [ref for ref in refs if ref.rsplit("/", 1)[-1] == DESKTOP_BRANCH]
    if any(is_ancestor("HEAD", ref) for ref in desktops):
        return True
    if branch is None or not desktops:
        return False
    variants = [ref for ref in refs if f"/{VARIANT_PREFIX}" in ref]
    return ahead_of(desktops) <= ahead_of(variants)


def ahead_of(refs: list[str]) -> float:
    """How many commits HEAD has that the nearest of `refs` lacks."""
    counts = [git("rev-list", "--count", f"{ref}..HEAD") for ref in refs]
    return min((int(count) for count in counts if count), default=float("inf"))


def upstream_ref() -> str | None:
    """Upstream's dev branch, found by the remote's URL: the remote is
    `origin` in one checkout and `upstream` in the next."""
    remotes = (git("remote") or "").split()
    for remote in remotes:
        url = git("remote", "get-url", remote) or ""
        ref = f"refs/remotes/{remote}/{UPSTREAM_BRANCH}"
        if UPSTREAM.search(url) and git("rev-parse", "--verify", "--quiet", ref):
            return ref
    return None


def newest_upstream_in_head(ref: str) -> str | None:
    """The newest commit of upstream's that HEAD contains.

    Normally the merge base with upstream's branch. The local copy of that
    branch can be older than what the branch has merged, though: the daily
    sync's merge arrives from the fork, and nothing fetches upstream. The
    merge base is then the stale ref itself, and everything upstream changed
    since would count as the fork's. The newer upstream commit is the second
    parent of the merge that brought it in; upstream's commits are the ones
    without the desktop directory."""
    base = git("merge-base", "HEAD", ref)
    if base is None or base != git("rev-parse", ref):
        return base
    merges = git("rev-list", "--merges", "--parents", f"{base}..HEAD") or ""
    for line in merges.splitlines():  # newest first
        for parent in line.split()[2:]:
            theirs = git("cat-file", "-e", f"{parent}:{OURS.rstrip('/')}") is None
            if theirs and is_ancestor(base, parent):
                return parent
    return base


def fork_surface(environment: Mapping[str, str]) -> tuple[list[str], str, str]:
    """(what differs from upstream outside the desktop's own, the upstream
    ref, the upstream commit compared with). Skips when the rule is not this
    checkout's, or cannot be checked here."""
    if git("rev-parse", "--git-dir") is None:
        pytest.skip("not a git checkout (or git is not installed)")
    branch = git("symbolic-ref", "--short", "-q", "HEAD")
    applies, reason = rule_applies(environment, branch, head_is_in_desktop(branch))
    if not applies:
        pytest.skip(reason)
    ref = upstream_ref()
    if ref is None:
        pytest.skip(
            "no remote of Significant-Gravitas/AutoGPT with its dev branch fetched; the "
            "upstream sync, which has it, is where this bites"
        )
    base = newest_upstream_in_head(ref)
    if base is None:
        pytest.skip(f"no common ancestor with {ref} (a shallow checkout)")
    # Against the working tree, so that it bites before the commit; a new file
    # is in no diff until it is added.
    changed = git("diff", "--name-only", "--no-renames", base, "--")
    untracked = git("ls-files", "--others", "--exclude-standard")
    assert (
        changed is not None and untracked is not None
    ), f"git diff against {base} failed"
    return foreign([*changed.splitlines(), *untracked.splitlines()]), ref, base


def test_the_fork_changes_nothing_outside_the_desktop_directory():
    outside, ref, base = fork_surface(os.environ)
    assert not outside, (
        f"the fork differs from upstream ({ref}, at {base[:12]}) outside {OURS}: {outside}. "
        "Upstream edits these files, so each is a merge conflict waiting for the daily sync. "
        f"Restore upstream's content (git show {base[:12]}:<path>; delete a file upstream does "
        "not have), do what the change did from inside autogpt_platform/desktop, and offer "
        "the change upstream as a patch in autogpt_platform/desktop/upstream/."
    )


# --- the same, on a repository made for the purpose ---------------------------------


class Scratch:
    """A fork in miniature: upstream's branch as a remote ref, and `desktop`
    on top of it with the desktop directory."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        root.mkdir()
        monkeypatch.setattr(sys.modules[__name__], "REPO", root)
        self.do("init", "-q", "-b", "dev")
        self.do(
            "remote", "add", "sg", "https://github.com/Significant-Gravitas/AutoGPT.git"
        )
        self.first = self.commit("autogpt_platform/single-container/README.md", "one")
        self.track(self.first)
        self.do("checkout", "-q", "-b", DESKTOP_BRANCH)
        self.commit(f"{OURS}README.md", "the desktop app")

    def do(self, *arguments: str) -> str:
        # The developer's own git configuration (signing, line endings) has
        # no say in a repository made for a test.
        environment = {
            **os.environ,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
        }
        result = subprocess.run(
            ["git", "-C", str(self.root), *arguments],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            env=environment,
            timeout=120,
            check=True,
        )
        return result.stdout.strip()

    def write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def commit(self, relative: str, text: str) -> str:
        self.write(relative, text)
        self.do("add", "-A")
        self.do("commit", "-q", "-m", f"change {relative}")
        return self.do("rev-parse", "HEAD")

    def track(self, commit: str) -> None:
        self.do("update-ref", f"refs/remotes/sg/{UPSTREAM_BRANCH}", commit)

    def upstream_moves_on(self) -> str:
        """Upstream changes its own file; returns the new commit, which no
        local ref points at yet."""
        self.do("checkout", "-q", "--detach", self.first)
        newer = self.commit("autogpt_platform/single-container/README.md", "two")
        self.do("checkout", "-q", DESKTOP_BRANCH)
        return newer

    def merge(self, commit: str) -> None:
        self.do(
            "merge", "-q", "--no-ff", "--no-edit", "-m", "Merge upstream dev", commit
        )


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Scratch:
    if run_git("--version") is None:
        pytest.skip("git is not installed")
    return Scratch(tmp_path / "fork", monkeypatch)


def test_a_clean_fork_passes_and_a_changed_upstream_file_does_not(scratch: Scratch):
    outside, ref, base = fork_surface({})
    assert (outside, ref, base) == ([], "refs/remotes/sg/dev", scratch.first)
    scratch.write("autogpt_platform/single-container/README.md", "ours")
    assert fork_surface({})[0] == ["autogpt_platform/single-container/README.md"]


def test_a_new_file_outside_the_desktop_directory_counts_before_it_is_added(
    scratch: Scratch,
):
    scratch.write("autogpt_platform/notes.txt", "never added")
    scratch.write(f"{OURS}notes.txt", "never added either, and ours")
    assert fork_surface({})[0] == ["autogpt_platform/notes.txt"]


def test_what_the_sync_merged_is_not_the_forks_when_the_upstream_ref_is_stale(
    scratch: Scratch,
):
    """The sync's merge is pulled from the fork; nothing fetched upstream."""
    newer = scratch.upstream_moves_on()
    scratch.merge(newer)
    outside, _, base = fork_surface({})
    assert (outside, base) == ([], newer)
    scratch.track(newer)
    assert fork_surface({}) == ([], "refs/remotes/sg/dev", newer)


def test_a_merged_branch_of_the_forks_own_is_not_taken_for_upstream(scratch: Scratch):
    scratch.do("checkout", "-q", "-b", "some-work")
    scratch.commit("autogpt_platform/single-container/README.md", "ours, on a branch")
    scratch.do("checkout", "-q", DESKTOP_BRANCH)
    scratch.merge("some-work")
    outside, _, base = fork_surface({})
    assert (outside, base) == (
        ["autogpt_platform/single-container/README.md"],
        scratch.first,
    )


def test_a_branch_started_from_desktop_is_held_to_the_rule(scratch: Scratch):
    scratch.do("branch", "variant/voice")
    scratch.do("checkout", "-q", "-b", "some-work")
    scratch.commit(".gitignore", "node_modules")
    assert fork_surface({})[0] == [".gitignore"]


def test_a_branch_started_from_a_variant_is_not(scratch: Scratch):
    scratch.do("checkout", "-q", "-b", "variant/voice")
    scratch.commit("autogpt_platform/backend/voice.py", "what the variant is for")
    scratch.do("checkout", "-q", "-b", "voice-work")
    scratch.commit(f"{OURS}README.md", "more")
    with pytest.raises(
        pytest.skip.Exception, match="voice-work is not part of desktop"
    ):
        fork_surface({})
    with pytest.raises(
        pytest.skip.Exception, match="GITHUB_REF_NAME is the variant branch"
    ):
        fork_surface({"GITHUB_REF_NAME": "variant/voice"})


def test_a_checkout_without_upstreams_branch_is_skipped_with_the_reason(
    scratch: Scratch,
):
    scratch.do("update-ref", "-d", f"refs/remotes/sg/{UPSTREAM_BRANCH}")
    with pytest.raises(
        pytest.skip.Exception, match="no remote of Significant-Gravitas/AutoGPT"
    ):
        fork_surface({})


# --- the one workaround the rule has cost: runtime_config.py on Windows -----------------

# Every name `os` has on Windows: CPython 3.13.15, `sorted(dir(os))` without
# the dunder names. test_the_recorded_windows_os_is_the_real_one holds it true.
WINDOWS_OS_PYTHON = (3, 13)
WINDOWS_OS_NAMES = """
    DirEntry EX_OK F_OK GenericAlias Mapping MutableMapping O_APPEND O_BINARY O_CREAT O_EXCL
    O_NOINHERIT O_RANDOM O_RDONLY O_RDWR O_SEQUENTIAL O_SHORT_LIVED O_TEMPORARY O_TEXT O_TRUNC
    O_WRONLY P_DETACH P_NOWAIT P_NOWAITO P_OVERLAY P_WAIT PathLike R_OK SEEK_CUR SEEK_END
    SEEK_SET TMP_MAX W_OK X_OK _AddedDllDirectory _Environ _check_methods _execvpe _exists
    _exit _fspath _get_exports_list _walk_symlinks_as_files _wrap_close abc abort access
    add_dll_directory altsep chdir chmod close closerange cpu_count curdir defpath
    device_encoding devnull dup dup2 environ error execl execle execlp execlpe execv execve
    execvp execvpe extsep fchmod fdopen fsdecode fsencode fspath fstat fsync ftruncate
    get_blocking get_exec_path get_handle_inheritable get_inheritable get_terminal_size getcwd
    getcwdb getenv getlogin getpid getppid isatty kill lchmod linesep link listdir listdrives
    listmounts listvolumes lseek lstat makedirs mkdir name open pardir path pathsep pipe popen
    process_cpu_count putenv read readlink remove removedirs rename renames replace rmdir
    scandir sep set_blocking set_handle_inheritable set_inheritable spawnl spawnle spawnv
    spawnve st startfile stat stat_result statvfs_result strerror supports_bytes_environ
    supports_dir_fd supports_effective_ids supports_fd supports_follow_symlinks symlink sys
    system terminal_size times times_result truncate umask uname_result unlink unsetenv
    urandom utime waitpid waitstatus_to_exitcode walk write
"""
WINDOWS_OS = frozenset(WINDOWS_OS_NAMES.split())
# Windows has had it only since Python 3.13; the app must not need it.
NEWER_THAN_SUPPORTED = {"fchmod"}


class LikeWindows:
    """`os` as Windows has it, on any system: nothing Windows lacks, no
    fchmod (new there in Python 3.13), and directories that cannot be
    opened. What is left is this system's own."""

    name = "nt"

    def __getattr__(self, name: str):
        if name not in WINDOWS_OS or name in NEWER_THAN_SUPPORTED:
            raise AttributeError(
                f"{RUNTIME_CONFIG} uses os.{name}, which Windows does not have. Teach "
                "WindowsOs in desktop/runtime/autogpt_desktop/settings.py to stand in for it, "
                f"and add the guard to {PATCH}."
            )
        return getattr(os, name)

    def open(self, path, flags: int, mode: int = 0o777) -> int:
        if os.path.isdir(path):
            raise PermissionError(13, "Permission denied", str(path))
        return os.open(path, flags, mode)


@pytest.fixture
def data(tmp_path: Path) -> DataDir:
    data = DataDir(tmp_path / "data")
    data.prepare()
    return data


def test_upstreams_runtime_config_runs_unmodified_on_this_system(
    tmp_path: Path, data: DataDir
):
    """The real module, the real OS: Windows, macOS and Linux each run this."""
    bundle = Bundle(
        tmp_path / "no-bundle"
    )  # so the source tree's copy is the one loaded
    first = settings.ensure_secrets(bundle, data)
    assert first == settings.ensure_secrets(bundle, data)
    assert data.runtime_env.read_text(encoding="ascii").count("\n") > 5
    leftovers = [
        path.name for path in data.config.iterdir() if path.name.endswith(".tmp")
    ]
    assert not leftovers


def test_upstreams_runtime_config_runs_where_windows_refuses(tmp_path: Path):
    """The same module with Windows's `os`, on any system: this is the one
    the upstream sync runs, on Linux. Anything upstream starts to use that
    Windows lacks stops the sync here, not a Windows build afterwards."""
    module = settings.load_runtime_config_module(Bundle(tmp_path))
    assert vars(module).get("os") is not None, (
        "single-container/runtime_config.py no longer does `import os`, so "
        "settings.WindowsOs has nothing to stand in for. See how it reaches fchmod and the "
        "directory fsync now, and update load_runtime_config_module."
    )
    vars(module)["os"] = settings.WindowsOs(LikeWindows())
    path = tmp_path / "config" / "runtime.env"
    first = module.ensure_runtime_config(path, {})
    assert first == module.ensure_runtime_config(path, {})
    assert [entry.name for entry in path.parent.iterdir()] == ["runtime.env"]


def test_the_stand_in_refuses_what_windows_refuses(tmp_path: Path):
    windows = LikeWindows()
    for missing in (
        "fchmod",
        "fchown",
        "geteuid",
        "getuid",
        "chown",
        "O_NOFOLLOW",
        "O_CLOEXEC",
    ):
        assert not hasattr(windows, missing), missing
    with pytest.raises(
        AttributeError, match=r"uses os\.fchown, which Windows does not have"
    ):
        windows.fchown(0, 0, 0)
    with pytest.raises(PermissionError):
        windows.open(tmp_path, os.O_RDONLY)
    assert windows.name == "nt" and windows.sep == os.sep


def test_the_recorded_windows_os_is_the_real_one():
    if sys.platform != "win32":
        pytest.skip("only Windows can say what its `os` has")
    if sys.version_info[:2] < WINDOWS_OS_PYTHON:
        pytest.skip(
            f"recorded from Python {WINDOWS_OS_PYTHON[0]}.{WINDOWS_OS_PYTHON[1]}"
        )
    real = {name for name in dir(os) if not name.startswith("__")}
    assert real == WINDOWS_OS and sys.version_info[:2] == WINDOWS_OS_PYTHON, (
        f"`os` on Windows Python {sys.version_info[0]}.{sys.version_info[1]} has "
        f"{sorted(real - WINDOWS_OS)} more and {sorted(WINDOWS_OS - real)} fewer names than "
        "WINDOWS_OS records. Record them (the comment above it says how) with WINDOWS_OS_PYTHON."
    )


def test_windows_os_leaves_files_alone_and_skips_only_the_directory(tmp_path: Path):
    synced: list[int] = []
    closed: list[int] = []
    real = SimpleNamespace(
        path=os.path,
        devnull=os.devnull,
        O_RDONLY=os.O_RDONLY,
        open=os.open,
        fsync=synced.append,
        close=lambda descriptor: (closed.append(descriptor), os.close(descriptor)),
        sep="/",
    )
    shim = settings.WindowsOs(real)
    file = tmp_path / "file"
    file.write_text("x", encoding="ascii")

    descriptor = shim.open(file, os.O_RDONLY)
    shim.fchmod(descriptor, 0o600)  # the real one has none: nothing to call
    shim.fsync(descriptor)
    shim.close(descriptor)
    assert synced == [descriptor] and closed == [descriptor]

    directory = shim.open(tmp_path, os.O_RDONLY)
    shim.fsync(directory)
    shim.close(directory)
    assert synced == [descriptor], "the directory's stand-in must not be fsynced"
    assert closed == [descriptor, directory], "and must be closed"
    assert shim.sep == "/", "everything else is the real module's"


def test_the_patch_offered_upstream_is_still_wanted_and_still_applies():
    patch = (REPO / PATCH).read_text(encoding="utf-8")
    guards = ('hasattr(os, "fchmod")', 'os.name == "nt"')
    assert all(guard in patch for guard in guards)
    assert f"--- a/{RUNTIME_CONFIG}" in patch
    source = (REPO / RUNTIME_CONFIG).read_text(encoding="utf-8")
    assert not all(guard in source for guard in guards), (
        f"upstream has the fix ({RUNTIME_CONFIG} guards fchmod and the directory fsync "
        "itself). Delete WindowsOs from desktop/runtime/autogpt_desktop/settings.py, its "
        f"tests here, {PATCH} and its row in desktop/upstream/README.md."
    )
    if git("rev-parse", "--git-dir") is None:
        pytest.skip(
            "not a git checkout (or git is not installed): cannot try the patch"
        )
    result = run_git("apply", "--check", PATCH)
    assert result is not None and result.returncode == 0, (
        f"upstream changed {RUNTIME_CONFIG} (or its test) near the Windows fix, and {PATCH} "
        "no longer applies. See that WindowsOs still covers what the file does, then make the "
        "patch again from the new file (desktop/upstream/README.md).\n"
        f"{result.stderr if result else ''}"
    )
