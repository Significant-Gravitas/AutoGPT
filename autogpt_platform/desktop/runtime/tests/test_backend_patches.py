"""The backend is changed in the bundle's copy, and nowhere else.

build/backend_patches.py replaces exact pieces of the backend's text while a
bundle is built. These tests try every patch on the backend in the
repository, so that an upstream change to those lines stops the unit tests
(and the daily upstream sync that runs them) and not a build days later; and
they hold each patch equal to the diff offered upstream for it
(desktop/upstream/), so the bundle runs what upstream is asked to merge.
"""

import importlib
import ntpath
import posixpath
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

DESKTOP = Path(__file__).resolve().parents[2]
REPO = DESKTOP.parents[1]
BACKEND = DESKTOP.parent / "backend"
UPSTREAM = DESKTOP / "upstream"
sys.path.insert(0, str(DESKTOP / "build"))
backend_patches = importlib.import_module("backend_patches")

PATCHES = backend_patches.PATCHES
by_name = pytest.mark.parametrize("patch", PATCHES, ids=lambda patch: patch.name)


def source(patch) -> str:
    return backend_patches.read(BACKEND / patch.file)


def git(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


# --- against the backend in the repository -------------------------------------------


@by_name
def test_the_patch_still_fits_the_backend(patch):
    text = source(patch)
    assert not patch.is_applied(text), (
        f"upstream has the change {patch.name!r} makes (backend/{patch.file}). Delete the patch "
        f"from desktop/build/backend_patches.py, desktop/upstream/{patch.name}.patch and its row "
        "in desktop/upstream/README.md."
    )
    patched = patch.apply(text)  # says what to do when it does not fit
    assert patch.is_applied(patched)
    compile(patched, patch.file, "exec")


@by_name
def test_the_patch_is_offered_upstream_and_is_the_same_change(patch, tmp_path: Path):
    offered = UPSTREAM / f"{patch.name}.patch"
    assert offered.is_file(), f"desktop/upstream/{patch.name}.patch is missing"
    text = offered.read_text(encoding="utf-8")
    relative = f"autogpt_platform/backend/{patch.file}"
    assert f"--- a/{relative}" in text
    assert f"`{patch.name}.patch`" in (UPSTREAM / "README.md").read_text(
        encoding="utf-8"
    )
    if git("--version", cwd=tmp_path) is None:
        pytest.skip("git is not installed: cannot try the patch")

    # A copy of the files the diff names, outside any repository.
    for name in re.findall(r"^--- a/(\S+)$", text, re.M):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            backend_patches.read(REPO / name), encoding="utf-8", newline="\n"
        )
    shutil.copy2(offered, tmp_path / "offered.patch")
    applied = git("apply", "offered.patch", cwd=tmp_path)
    assert applied is not None and applied.returncode == 0, (
        f"upstream changed backend/{patch.file} near the lines of desktop/upstream/"
        f"{patch.name}.patch, which no longer applies. Make the patch again from the new file "
        f"(desktop/upstream/README.md).\n{applied.stderr if applied else ''}"
    )
    upstreams = backend_patches.read(tmp_path / relative)
    assert upstreams == patch.apply(source(patch)), (
        f"desktop/upstream/{patch.name}.patch and its entry in desktop/build/backend_patches.py "
        "no longer make the same change"
    )


# --- the mechanism --------------------------------------------------------------------


def test_a_patch_refuses_text_that_is_missing_or_there_twice():
    patch = backend_patches.Patch(
        "example", "backend/x.py", "old = 1\n", "new = 1\n", "why"
    )
    assert patch.apply("a\nold = 1\nb\n") == "a\nnew = 1\nb\n"
    for text in ("a\nb\n", "old = 1\nold = 1\n"):
        with pytest.raises(ValueError, match="no longer fits backend/backend/x.py"):
            patch.apply(text)


def test_the_build_patches_a_copy_and_the_seal_notices_one_that_was_not(tmp_path: Path):
    for patch in PATCHES:
        copy = tmp_path / patch.file
        copy.parent.mkdir(parents=True, exist_ok=True)
        # As a Windows checkout has it.
        copy.write_bytes(source(patch).replace("\n", "\r\n").encode("utf-8"))
    with pytest.raises(RuntimeError, match="lacks the build-time patches"):
        backend_patches.check(tmp_path)

    backend_patches.apply(tmp_path)

    backend_patches.check(tmp_path)
    for patch in PATCHES:
        assert b"\r\n" not in (tmp_path / patch.file).read_bytes()
    with pytest.raises(ValueError, match="no longer fits"):
        backend_patches.apply(tmp_path)  # a second time: the build copies afresh


def test_the_build_patches_before_it_compiles():
    build = (DESKTOP / "build" / "build_runtime.py").read_text(encoding="utf-8")
    steps = build[build.index("STEPS = (") :]
    assert steps.index('"backend"') < steps.index('"compile"')
    assert "backend_patches.apply(target)" in build
    assert "backend_patches.check(" in build


# --- what the workspace patch does, on either kind of system --------------------------


WORKSPACE_PATCH = "copilot-workspace-prefix"


def workspace_patch():
    """The tests below are about this one patch, and go when it goes."""
    found = [patch for patch in PATCHES if patch.name == WORKSPACE_PATCH]
    assert found, (
        f"the {WORKSPACE_PATCH} patch is gone from desktop/build/backend_patches.py. If "
        "upstream merged it, delete the tests under 'what the workspace patch does' in "
        "desktop/runtime/tests/test_backend_patches.py too (desktop/upstream/README.md "
        "lists them)."
    )
    return found[0]


def make_session_path(path_module, prefix: str | None):
    """sandbox.make_session_path of the patched backend, as it would be on a
    system with `path_module` for os.path."""
    patch = workspace_patch()
    patched = patch.apply(source(patch))
    constant = patched[patched.index(patch.replacement) :].split("\n\n\n", 1)[0]
    function = patched[patched.index("def make_session_path(") :].split("\n\n\n", 1)[0]
    environ = {"COPILOT_WORKSPACE_PREFIX": prefix} if prefix else {}
    scope = {"os": SimpleNamespace(path=path_module, environ=environ)}
    exec(f"{constant}\n{function}", scope)
    return scope["WORKSPACE_PREFIX"], scope["make_session_path"]


def test_the_workspace_prefix_is_unchanged_on_posix():
    prefix, session_path = make_session_path(posixpath, None)
    assert prefix == "/tmp/copilot-"
    assert session_path("abc-123") == "/tmp/copilot-abc-123"


def test_a_session_path_can_be_built_on_windows():
    """Unpatched, this raises "Session path escaped prefix"."""
    _, session_path = make_session_path(ntpath, None)
    assert session_path("abc-123") == "\\tmp\\copilot-abc-123"
    prefix, session_path = make_session_path(
        ntpath, "C:\\Users\\x\\AppData\\AutoGPT\\tmp\\copilot-"
    )
    assert session_path("abc-123") == f"{prefix}abc-123"
    assert session_path("../../etc") == f"{prefix}etc"


def test_unpatched_a_session_path_cannot_be_built_on_windows():
    """What the patch is for. When this stops raising, upstream fixed it."""
    patch = workspace_patch()
    text = source(patch)
    function = text[text.index("def make_session_path(") :].split("\n\n\n", 1)[0]
    scope = {"os": SimpleNamespace(path=ntpath)}
    exec(f"{patch.original}\n{function}", scope)
    with pytest.raises(ValueError, match="escaped prefix"):
        scope["make_session_path"]("abc-123")


def test_every_check_of_a_workspace_path_reads_the_patched_prefix():
    """The three places that refuse a path outside the prefix (building it,
    using it as the SDK's cwd, cleaning it up) compare with one constant."""
    workspace_patch()
    sandbox = backend_patches.read(
        BACKEND / "backend" / "copilot" / "tools" / "sandbox.py"
    )
    service = backend_patches.read(
        BACKEND / "backend" / "copilot" / "sdk" / "service.py"
    )
    update = (
        "Read how the backend checks a workspace path now, and extend the "
        "copilot-workspace-prefix patch (desktop/build/backend_patches.py) to every check."
    )
    assert "if not path.startswith(WORKSPACE_PREFIX):" in sandbox, update
    assert "_SDK_CWD_PREFIX = WORKSPACE_PREFIX" in service, update
    assert service.count(".startswith(_SDK_CWD_PREFIX)") == 2, update
    literal = re.compile(r'^[^#\n]*=\s*.*"/tmp/copilot-"', re.M)
    others = [
        str(path.relative_to(BACKEND))
        for path in sorted((BACKEND / "backend" / "copilot").rglob("*.py"))
        if not path.name.endswith("_test.py")
        and literal.search(backend_patches.read(path))
    ]
    # transcript.py: _SAFE_CWD_PREFIX, used only by write_transcript_to_tempfile,
    # which nothing calls.
    expected = [
        str(Path("backend/copilot/tools/sandbox.py")),
        str(Path("backend/copilot/transcript.py")),
    ]
    assert (
        others == expected
    ), f"the backend now builds a path from the literal /tmp/copilot- in {others}. {update}"
    # Every mention outside the tests, by file: its definition and its
    # re-export. One more anywhere, the defining file included, is a caller.
    mentions = {
        path.relative_to(BACKEND)
        .as_posix(): backend_patches.read(path)
        .count("write_transcript_to_tempfile")
        for path in sorted((BACKEND / "backend").rglob("*.py"))
        if not path.name.endswith("_test.py") and not path.name.startswith("test_")
    }
    mentions = {name: count for name, count in mentions.items() if count}
    assert mentions == {
        "backend/copilot/sdk/transcript.py": 2,
        "backend/copilot/transcript.py": 1,
    }, (
        f"write_transcript_to_tempfile is used now ({mentions}), and checks its directory "
        f"against a prefix of its own (transcript._SAFE_CWD_PREFIX). {update}"
    )


def test_the_model_is_told_its_real_working_directory_each_turn():
    """The system prompt names /tmp/copilot-<session-id> on every system: a
    placeholder, kept constant for the prompt cache. What the model works
    from is the directory in the first message of the turn, which is the real
    one, on Windows the one in the data directory."""
    workspace_patch()
    prompting = backend_patches.read(BACKEND / "backend" / "copilot" / "prompting.py")
    service = backend_patches.read(
        BACKEND / "backend" / "copilot" / "sdk" / "service.py"
    )
    update = (
        "Read how the model learns its working directory now (backend/copilot/prompting.py, "
        "sdk/service.py). If the system prompt's /tmp/copilot-<session-id> is all it is told, "
        "that is wrong on Windows: make prompting.py read sandbox.WORKSPACE_PREFIX with one "
        "more patch in desktop/build/backend_patches.py."
    )
    assert prompting.count('"/tmp/copilot-') == 1, update
    assert (
        '_get_local_storage_supplement("/tmp/copilot-<session-id>")' in prompting
    ), update
    assert "generic placeholder for the working directory" in prompting, update
    assert 'env_ctx_content = f"working_dir: {sdk_cwd}"' in service, update
