"""The Prisma CLI is run with the bundle's engines and nothing to fetch.

What the CLI does with these variables was read in Prisma 5.17
(@prisma/engines ensureBinariesExist, @prisma/fetch-engine download). The
build checks that the CLI it bundles still reads them, and the smoke test
runs the migrations with the network out of reach; here, only what the
runtime hands the CLI."""

import subprocess
import sys
from pathlib import Path

import pytest

from autogpt_desktop import bootstrap, migrations
from autogpt_desktop.layout import EXE, Bundle


@pytest.fixture
def bundle(tmp_path: Path) -> Bundle:
    return Bundle(tmp_path / "runtime")


def test_the_cli_is_pointed_at_the_bundled_engines_as_executables(bundle: Bundle):
    env = migrations.environment(bundle, {"DATABASE_URL": "postgresql://x"})

    assert env["PRISMA_CLI_QUERY_ENGINE_TYPE"] == "binary"
    assert env["PRISMA_QUERY_ENGINE_BINARY"] == str(
        bundle.root / "prisma" / f"query-engine{EXE}"
    )
    assert env["PRISMA_SCHEMA_ENGINE_BINARY"] == str(
        bundle.root / "prisma" / f"schema-engine{EXE}"
    )
    assert env["DATABASE_URL"] == "postgresql://x"
    assert env["CHECKPOINT_DISABLE"] == "1"
    assert set(migrations.READ_BY_THE_CLI) <= env.keys()


def test_prisma_settings_from_the_users_own_shell_do_not_reach_the_cli(
    bundle: Bundle, monkeypatch
):
    monkeypatch.setenv("PRISMA_ENGINES_MIRROR", "https://mirror.example")
    monkeypatch.setenv("PRISMA_CLI_BINARY_TARGETS", "debian-openssl-1.1.x")
    monkeypatch.setenv(
        "PRISMA_QUERY_ENGINE_LIBRARY", "/somewhere/libquery_engine.so.node"
    )
    monkeypatch.setenv("PRISMA_CLI_QUERY_ENGINE_TYPE", "library")
    monkeypatch.setenv("SOMETHING_ELSE", "kept")

    env = migrations.environment(bundle, {"PRISMA_SCHEMA": "set by the runtime"})

    assert "PRISMA_ENGINES_MIRROR" not in env
    assert "PRISMA_CLI_BINARY_TARGETS" not in env
    assert "PRISMA_QUERY_ENGINE_LIBRARY" not in env
    assert env["PRISMA_CLI_QUERY_ENGINE_TYPE"] == "binary"
    assert env["PRISMA_SCHEMA"] == "set by the runtime"
    assert env["SOMETHING_ELSE"] == "kept"


def test_a_missing_openssl_is_named_instead_of_an_engine_error():
    loader = (
        "/opt/AutoGPT/resources/runtime/prisma/schema-engine: error while loading shared "
        "libraries: libssl.so.3: cannot open shared object file: No such file or directory"
    )
    assert migrations.engine_problem(127, loader) == migrations.OPENSSL_MISSING
    assert "libssl.so.3" in migrations.OPENSSL_MISSING
    crypto = loader.replace("libssl.so.3", "libcrypto.so.3")
    assert migrations.engine_problem(127, crypto) == migrations.OPENSSL_MISSING
    other = loader.replace("libssl.so.3", "libz.so.1")
    assert "libz.so.1" in str(migrations.engine_problem(127, other))
    # An engine built against another OpenSSL (the build refuses to bundle
    # one) is not answered with advice to install the one the system has.
    wrong_build = migrations.engine_problem(
        127, loader.replace("libssl.so.3", "libssl.so.1.1")
    )
    assert "libssl.so.1.1" in str(wrong_build) and "OpenSSL 3" not in str(wrong_build)
    assert migrations.engine_problem(0, "") is None
    assert migrations.engine_problem(1, "some other failure") is None


def test_migrations_do_not_start_on_a_linux_without_openssl_3(
    bundle: Bundle, monkeypatch
):
    calls = []

    def run_tool(argv: list[str], **kwargs) -> subprocess.CompletedProcess[bytes]:
        calls.append(argv)
        stderr = b"schema-engine: error while loading shared libraries: libssl.so.3: cannot open"
        return subprocess.CompletedProcess(argv, 127, b"", stderr)

    monkeypatch.setattr(migrations.sys, "platform", "linux")
    monkeypatch.setattr(migrations, "run_tool", run_tool)
    monkeypatch.setattr(bootstrap, "run_tool", run_tool)

    with pytest.raises(RuntimeError, match="OpenSSL 3"):
        bootstrap.apply_migrations(bundle, {})

    assert calls == [[str(bundle.prisma_engine("schema-engine")), "--version"]]


def test_migrations_run_the_cli_with_that_environment(bundle: Bundle, monkeypatch):
    seen = {}

    def run_tool(argv: list[str], **kwargs) -> subprocess.CompletedProcess[bytes]:
        seen.update(argv=argv, env=kwargs["env"])
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(
        migrations.sys,
        "platform",
        sys.platform if sys.platform != "linux" else "darwin",
    )
    monkeypatch.setattr(bootstrap, "run_tool", run_tool)

    bootstrap.apply_migrations(bundle, {"DATABASE_URL": "postgresql://x"})

    assert seen["argv"][-4:-2] == ["migrate", "deploy"]
    assert seen["env"] == migrations.environment(
        bundle, {"DATABASE_URL": "postgresql://x"}
    )


def test_an_engine_that_cannot_even_be_run_is_left_to_the_migration_to_report(
    bundle: Bundle, monkeypatch
):
    def run_tool(argv: list[str], **kwargs) -> subprocess.CompletedProcess[bytes]:
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(migrations.sys, "platform", "linux")
    monkeypatch.setattr(migrations, "run_tool", run_tool)
    migrations.require_engines(bundle)
