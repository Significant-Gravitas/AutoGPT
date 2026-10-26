"""What the build and its CI assume about files they do not own.

The in-repository workflow checks the selected source on every supported
test platform. Native packaging is an explicit opt-in and never publishes.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

DESKTOP = Path(__file__).resolve().parents[2]
PLATFORM = DESKTOP.parent
REPO = PLATFORM.parent
CALLER = REPO / ".github" / "workflows" / "platform-desktop-build.yml"

# The frontend's own build script. The desktop build has to do the same
# things in the same order, but cannot run the script as it is on a machine
# with less than 16 GB (build_runtime.step_frontend).
FRONTEND_BUILD_SCRIPT = (
    "pnpm run copy:vad-assets && "
    "cross-env NODE_OPTIONS=--max-old-space-size=16384 next build"
)


def build_artifacts() -> ModuleType:
    """build/artifacts.py, which is a script directory's module, not a package's."""
    spec = importlib.util.spec_from_file_location(
        "desktop_build_artifacts", DESKTOP / "build" / "artifacts.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


def test_the_frontend_build_script_is_the_one_the_desktop_build_mirrors():
    scripts = json.loads((PLATFORM / "frontend" / "package.json").read_text("utf-8"))[
        "scripts"
    ]
    assert scripts["build"] == FRONTEND_BUILD_SCRIPT, (
        "upstream changed the frontend's `build` script. Make "
        "build/build_runtime.py step_frontend do what the new script does, "
        "then update FRONTEND_BUILD_SCRIPT here."
    )
    assert (
        "copy:vad-assets" in scripts
    ), "the build's first half is gone from frontend/package.json"


def test_workflow_builds_selected_source_without_fork_workflow_or_secrets():
    workflow = CALLER.read_text("utf-8")
    assert "uses: actions/checkout@" in workflow
    assert "ref:" not in workflow
    assert "ntindle/autogpt/" not in workflow
    assert "secrets:" not in workflow and "secrets." not in workflow
    assert "contents: read" in workflow
    assert "persist-credentials: false" in workflow
    assert "python -B build/build_runtime.py" in workflow
    assert "npm run dist -- --publish never" in workflow


def test_packaging_requires_explicit_opt_in_and_uses_isolated_identity():
    workflow = CALLER.read_text("utf-8")
    assert "branches: [dev]" in workflow
    assert "default: false" in workflow
    assert "github.event_name == 'workflow_dispatch' && inputs.package" in workflow
    assert (
        "github.event.pull_request.head.repo.full_name == github.repository" in workflow
    )
    assert (
        "contains(github.event.pull_request.labels.*.name, 'desktop-build')" in workflow
    )
    assert "needs: checks" in workflow
    assert "AUTOGPT_DESKTOP_VARIANT: experimental" in workflow
    assert "AUTOGPT_DESKTOP_VERSION: 0.0.0-dev." in workflow


def test_every_path_that_triggers_the_build_exists():
    """A path filter naming a file upstream renamed never triggers again."""
    paths = re.findall(r'^\s+- "([^"]+)"$', CALLER.read_text("utf-8"), re.M)
    assert set(paths) == {
        ".github/workflows/platform-desktop-build.yml",
        "autogpt_platform/desktop/**",
        "autogpt_platform/backend/**",
        "autogpt_platform/autogpt_libs/**",
        "autogpt_platform/frontend/**",
        "autogpt_platform/single-container/**",
        "autogpt_platform/db/init/**",
    }
    missing = [path for path in paths if not (REPO / path.removesuffix("/**")).exists()]
    assert (
        not missing
    ), f"the build is triggered by paths that no longer exist: {missing}"


def test_valkey_for_windows_is_built_from_the_archive_the_macos_pin_names():
    """valkey-windows.sh is given the macOS entry's digest, which is only
    right while both download the same file."""
    artifacts = build_artifacts()
    script = (DESKTOP / "build" / "valkey-windows.sh").read_text("utf-8")
    url = re.search(r'curl [^\n]*"(https://[^"]+)"', script)
    assert url, "valkey-windows.sh no longer downloads with curl"
    expected = url.group(1).replace("${version}", artifacts.VALKEY_VERSION)
    assert artifacts.ARTIFACTS["darwin-arm64"]["valkey"].url == expected
    assert "sha256sum -c" in script


def test_the_lockfiles_match_their_manifests():
    """`npm ci` refuses a lockfile that disagrees with package.json."""
    for directory in (DESKTOP, DESKTOP / "e2e"):
        manifest = json.loads((directory / "package.json").read_text("utf-8"))
        lock = json.loads((directory / "package-lock.json").read_text("utf-8"))
        root = lock["packages"][""]
        assert root.get("devDependencies") == manifest.get("devDependencies"), directory
        assert lock["name"] == manifest["name"], directory
        for name, version in manifest["devDependencies"].items():
            assert re.fullmatch(
                r"\d+\.\d+\.\d+", version
            ), f"{name} is not pinned exactly"
            assert lock["packages"][f"node_modules/{name}"]["version"] == version
