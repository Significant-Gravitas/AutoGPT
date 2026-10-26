"""The bundle carries the locked claude-agent-sdk and the CLI it names
(build/claude_cli.py), on every platform.

The stand-in CLI is claude_stub.py's; the real one is never started, except
by the tests of an assembled bundle, which ask its CLI for its version.
"""

import importlib
import sys
from pathlib import Path

import claude_stub
import pytest

DESKTOP = Path(__file__).resolve().parents[2]
LOCK = DESKTOP.parent / "backend" / "poetry.lock"
REAL_BUNDLE = DESKTOP / "build" / "runtime"
sys.path.insert(0, str(DESKTOP / "build"))
artifacts = importlib.import_module("artifacts")
backend_patches = importlib.import_module("backend_patches")
build_runtime = importlib.import_module("build_runtime")
claude_cli = importlib.import_module("claude_cli")
lock_export = importlib.import_module("lock_export")

PLATFORMS = ("win32", "darwin", "linux")


@pytest.fixture
def sdk(tmp_path: Path) -> Path:
    """site-packages with an SDK installed from source: no _bundled."""
    packages = tmp_path / "site-packages"
    (packages / "claude_agent_sdk").mkdir(parents=True)
    (packages / "claude_agent_sdk-0.2.161.dist-info").mkdir()
    return packages / "claude_agent_sdk"


def declare(sdk: Path, version: str) -> None:
    (sdk / "_cli_version.py").write_text(
        f'"""Bundled Claude Code CLI version."""\n\n__cli_version__ = "{version}"\n',
        encoding="utf-8",
    )


@pytest.fixture
def published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A CLI "in the release bucket": `download` hands it over. Its version
    is the stand-in's own (claude_stub.version_of)."""
    cli = claude_stub.make(tmp_path / "published")
    monkeypatch.setattr(claude_cli, "download", lambda artifact, cache, name: cli)
    return cli


ARTIFACT = artifacts.Artifact("https://example.invalid/claude", "0" * 64)


def test_every_platform_installs_the_sdk_version_the_lock_pins():
    """What the Docker image runs. A platform without a wheel for it builds
    it from source; none gets another version."""
    locked = claude_cli.locked_sdk_version(LOCK)
    for platform in PLATFORMS:
        pins = dict(
            line.split(" ; ")[0].split("==")
            for line in lock_export.export(LOCK, platform)
            if "==" in line
        )
        assert pins[claude_cli.SDK] == locked, platform
    assert lock_export.PLATFORM_OVERRIDES == {}


def test_a_source_install_gets_the_cli_its_sdk_names(
    sdk: Path, published: Path, tmp_path: Path
):
    version = claude_stub.version_of(published, build_probe=True)
    declare(sdk, version)

    cli = claude_cli.ensure(sdk / "_bundled", ARTIFACT, version, tmp_path / "cache")

    assert cli == sdk / "_bundled" / claude_stub.NAME
    assert cli.read_bytes() == published.read_bytes(), "byte for byte as published"


def test_a_cli_the_wheel_brought_is_left_alone(
    sdk: Path, published: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    brought = claude_stub.make(sdk / "_bundled")
    version = claude_stub.version_of(brought, build_probe=True)
    declare(sdk, version)
    before = brought.stat().st_mtime_ns

    def no_download(artifact, cache, name):
        raise AssertionError("nothing to download")

    monkeypatch.setattr(claude_cli, "download", no_download)
    assert (
        claude_cli.ensure(sdk / "_bundled", None, version, tmp_path / "cache")
        == brought
    )
    assert brought.stat().st_mtime_ns == before


def test_a_stale_pin_stops_the_build_where_the_wheel_brought_the_cli_too(
    sdk: Path, tmp_path: Path
):
    """macOS and Linux download nothing, and would otherwise never notice
    that artifacts.py names the CLI of an older lock."""
    brought = claude_stub.make(sdk / "_bundled")
    declare(sdk, claude_stub.version_of(brought, build_probe=True))
    for artifact in (None, ARTIFACT):
        with pytest.raises(
            RuntimeError, match="build/artifacts.py pins 0.0.0.*CLAUDE_CLI_VERSION"
        ):
            claude_cli.ensure(sdk / "_bundled", artifact, "0.0.0", tmp_path / "cache")


def test_a_lock_bump_that_changes_the_cli_stops_the_build(
    sdk: Path, published: Path, tmp_path: Path
):
    declare(sdk, "9.9.9")
    with pytest.raises(RuntimeError) as refused:
        claude_cli.ensure(sdk / "_bundled", ARTIFACT, "2.1.284", tmp_path / "cache")
    message = str(refused.value)
    assert "built with Claude Code 9.9.9" in message and "pins 2.1.284" in message
    assert "CLAUDE_CLI_VERSION" in message
    assert not (sdk / "_bundled").exists()


def test_a_cli_left_from_an_older_sdk_is_replaced(
    sdk: Path, published: Path, tmp_path: Path
):
    """After a lock bump the old program is still in _bundled: the SDK
    installed from source brings none to overwrite it."""
    stale = sdk / "_bundled" / claude_stub.NAME
    stale.parent.mkdir()
    stale.write_bytes(b"not a program")
    version = claude_stub.version_of(published, build_probe=True)
    declare(sdk, version)

    claude_cli.ensure(sdk / "_bundled", ARTIFACT, version, tmp_path / "cache")

    assert stale.read_bytes() == published.read_bytes()


def test_a_platform_without_a_pinned_cli_needs_the_wheels(sdk: Path, tmp_path: Path):
    declare(sdk, "2.1.284")
    with pytest.raises(RuntimeError, match="brought none.*pin the official build"):
        claude_cli.ensure(sdk / "_bundled", None, "2.1.284", tmp_path / "cache")


def test_an_sdk_that_no_longer_names_its_cli_stops_the_build(sdk: Path, tmp_path: Path):
    with pytest.raises(RuntimeError, match="no longer says which Claude Code CLI"):
        claude_cli.ensure(sdk / "_bundled", ARTIFACT, "2.1.284", tmp_path / "cache")


def test_the_seal_refuses_another_sdk_than_the_locks(
    sdk: Path, published: Path, tmp_path: Path
):
    declare(
        sdk,
        claude_stub.version_of(claude_stub.make(sdk / "_bundled"), build_probe=True),
    )
    lock = tmp_path / "poetry.lock"

    version = claude_cli.declared_version(sdk)
    lock.write_text(
        '[[package]]\nname = "claude-agent-sdk"\nversion = "0.2.161"\n',
        encoding="utf-8",
    )
    claude_cli.check(sdk / "_bundled", lock, tmp_path / "cache", version)
    with pytest.raises(RuntimeError, match="build/artifacts.py pins 0.0.0"):
        claude_cli.check(sdk / "_bundled", lock, tmp_path / "cache", "0.0.0")

    lock.write_text(
        '[[package]]\nname = "claude-agent-sdk"\nversion = "0.2.170"\n',
        encoding="utf-8",
    )
    with pytest.raises(
        RuntimeError, match=r"has claude-agent-sdk 0\.2\.161; .* pins 0\.2\.170"
    ):
        claude_cli.check(sdk / "_bundled", lock, tmp_path / "cache", version)


def test_the_seal_refuses_a_missing_or_mismatched_cli(
    sdk: Path, published: Path, tmp_path: Path
):
    lock = tmp_path / "poetry.lock"
    lock.write_text(
        '[[package]]\nname = "claude-agent-sdk"\nversion = "0.2.161"\n',
        encoding="utf-8",
    )
    declare(sdk, "9.9.9")
    with pytest.raises(RuntimeError, match="has no Claude Code CLI"):
        claude_cli.check(sdk / "_bundled", lock, tmp_path / "cache", "9.9.9")
    claude_stub.make(sdk / "_bundled")
    with pytest.raises(RuntimeError, match="reports version .* built with 9.9.9"):
        claude_cli.check(sdk / "_bundled", lock, tmp_path / "cache", "9.9.9")


def test_the_pinned_cli_is_anthropics_own_build_of_the_pinned_version():
    for platform, pinned in artifacts.ARTIFACTS.items():
        cli = pinned.get("claude-cli")
        if cli is None:
            continue
        assert cli.url.startswith(
            "https://storage.googleapis.com/claude-code-dist-"
        ), platform
        assert (
            f"/claude-code-releases/{artifacts.CLAUDE_CLI_VERSION}/" in cli.url
        ), platform
        assert len(cli.sha256) == 64


def test_the_assembled_bundle_has_the_locked_sdk_and_its_cli(tmp_path: Path):
    """What the seal step checks, for a bundle that was put together or
    refreshed without it (`--skip seal`, `--only assets`)."""
    if not (REAL_BUNDLE / "manifest.json").is_file():
        pytest.skip("no assembled bundle (build/runtime)")
    claude_cli.check(
        REAL_BUNDLE / "site" / "claude_agent_sdk" / "_bundled",
        LOCK,
        tmp_path,
        artifacts.CLAUDE_CLI_VERSION,
    )


def test_the_assembled_bundles_backend_has_the_build_time_patches():
    """Without them no AutoPilot turn starts on Windows, and nothing else
    that runs a bundle sends one."""
    if not (REAL_BUNDLE / "manifest.json").is_file():
        pytest.skip("no assembled bundle (build/runtime)")
    backend_patches.check(REAL_BUNDLE / "backend")


# --- running the steps again on an assembled bundle ------------------------------------


@pytest.fixture
def build(tmp_path: Path):
    """A bundle whose interpreter has two packages installed, not relocated."""
    build = build_runtime.Build(tmp_path / "runtime", tmp_path / "cache")
    for package in ("alpha", "beta"):
        (build.site_packages / package).mkdir(parents=True)
        (build.site_packages / package / "__init__.py").write_text("", encoding="utf-8")
    return build


def packages(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


def test_relocating_twice_leaves_the_packages_where_they_are(build, capsys):
    relocation = build_runtime.RELOCATION_FILE
    build.step_relocate()
    assert packages(build.out / "site") == ["alpha", "beta"]
    assert packages(build.site_packages) == [relocation]

    build.step_relocate()

    assert packages(build.out / "site") == ["alpha", "beta"]
    assert packages(build.site_packages) == [relocation]
    assert "already" in capsys.readouterr().out


def test_relocating_never_replaces_the_bundles_packages_with_a_stray_one(build):
    """Somebody installed one package into a finished bundle; the seal step
    then says the packages are not relocated."""
    build.step_relocate()
    (build.site_packages / "stray").mkdir()

    with pytest.raises(
        RuntimeError, match=r"\['stray'\] were installed.*would replace"
    ):
        build.step_relocate()

    assert packages(build.out / "site") == [
        "alpha",
        "beta",
    ], "the bundle's packages are gone"
    with pytest.raises(RuntimeError, match="both .* hold packages"):
        build._restore_site_packages()
    assert packages(build.out / "site") == ["alpha", "beta"]


def test_a_fresh_install_replaces_an_older_relocation(build):
    """The python step starts the interpreter again; what an earlier build
    left in site/ is not this one's."""
    older = build.out / "site" / "gamma"
    older.mkdir(parents=True)

    build.step_relocate()

    assert packages(build.out / "site") == ["alpha", "beta"]


def test_the_deps_step_puts_relocated_packages_back_for_uv(build):
    build.step_relocate()

    build._restore_site_packages()

    assert packages(build.site_packages) == ["alpha", "beta"]
    assert not (build.out / "site").exists()
    build.step_relocate()
    assert packages(build.out / "site") == ["alpha", "beta"]


def test_the_seal_names_the_steps_that_relocate_safely():
    source = (DESKTOP / "build" / "build_runtime.py").read_text(encoding="utf-8")
    seal = source[source.index("def step_seal(") :]
    assert "run the deps step and then the relocate step" in seal
