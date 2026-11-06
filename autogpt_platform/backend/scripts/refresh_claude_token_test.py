"""refresh_claude_token.sh: prints no token, writes the .env privately.

Runs the real script with a fake ``claude login`` credentials file; every
token is fake.
"""

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).with_name("refresh_claude_token.sh")
_ACCESS = "sk-ant-oat01-FAKE-access-token-000000000000"
_REFRESH = "sk-ant-ort01-FAKE-refresh-token-11111111111"

pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("jq") is None or shutil.which("bash") is None,
    reason="Runs the script on a POSIX host with bash and jq",
)


@pytest.fixture
def host(tmp_path: Path) -> Path:
    """A home with a ``claude login`` credentials file."""
    claude = tmp_path / "home" / ".claude"
    claude.mkdir(parents=True)
    (claude / ".credentials.json").write_text(
        json.dumps(
            {"claudeAiOauth": {"accessToken": _ACCESS, "refreshToken": _REFRESH}}
        )
    )
    return tmp_path


def _run(host: Path, env_file: Path, *, path: str | None = None):
    env = {
        "HOME": str(host / "home"),
        "PATH": path or os.environ["PATH"],
        "LANG": "C",
    }
    return subprocess.run(
        [
            "bash",
            "-c",
            'umask 022; exec bash "$0" --env-file "$1"',
            str(_SCRIPT),
            str(env_file),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def _shows_a_token(result: subprocess.CompletedProcess) -> bool:
    output = result.stdout + result.stderr
    return any(token[:12] in output for token in (_ACCESS, _REFRESH)) or any(
        token[-12:] in output for token in (_ACCESS, _REFRESH)
    )


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_a_new_env_file_is_owner_only_and_no_token_is_printed(host: Path):
    env_file = host / "backend.env"

    result = _run(host, env_file)

    assert result.returncode == 0, result.stderr
    assert not _shows_a_token(result)
    assert _mode(env_file) == 0o600
    assert env_file.read_text().splitlines() == [
        f"CLAUDE_CODE_OAUTH_TOKEN={_ACCESS}",
        f"CLAUDE_CODE_REFRESH_TOKEN={_REFRESH}",
        "CHAT_PLATFORM_ROUTE=subscription",
    ]


def test_an_existing_env_file_is_updated_in_place_and_closed_to_others(host: Path):
    env_file = host / "backend.env"
    env_file.write_text(
        "OTHER=1\n"
        "# CLAUDE_CODE_OAUTH_TOKEN=example\n"
        "CLAUDE_CODE_OAUTH_TOKEN=old\n"
        "# CLAUDE_CODE_REFRESH_TOKEN=\n"
        "CHAT_USE_CLAUDE_CODE_SUBSCRIPTION=true\n"
        "LAST=2\n"
    )
    os.chmod(env_file, 0o644)

    result = _run(host, env_file)

    assert result.returncode == 0, result.stderr
    assert not _shows_a_token(result)
    assert _mode(env_file) == 0o600
    assert env_file.read_text().splitlines() == [
        "OTHER=1",
        "# CLAUDE_CODE_OAUTH_TOKEN=example",
        f"CLAUDE_CODE_OAUTH_TOKEN={_ACCESS}",
        f"CLAUDE_CODE_REFRESH_TOKEN={_REFRESH}",
        "LAST=2",
        "CHAT_PLATFORM_ROUTE=subscription",
    ]


def test_a_symbolic_link_in_its_place_is_refused(host: Path):
    target = host / "elsewhere.env"
    target.write_text("OTHER=1\n")
    env_file = host / "backend.env"
    env_file.symlink_to(target)

    result = _run(host, env_file)

    assert result.returncode != 0
    assert "symbolic link" in result.stderr
    assert not _shows_a_token(result)
    assert target.read_text() == "OTHER=1\n"


def test_an_interrupted_write_leaves_the_old_file_whole(host: Path):
    """A failure before the rename keeps the old file and its temporary twin
    is cleaned up."""
    env_file = host / "backend.env"
    env_file.write_text("OTHER=1\n")
    shims = host / "bin"
    shims.mkdir()
    failing_mv = shims / "mv"
    failing_mv.write_text("#!/bin/sh\nexit 1\n")
    failing_mv.chmod(0o755)

    result = _run(host, env_file, path=f"{shims}:{os.environ['PATH']}")

    assert result.returncode != 0
    assert env_file.read_text() == "OTHER=1\n"
    assert sorted(p.name for p in host.iterdir()) == ["backend.env", "bin", "home"]
