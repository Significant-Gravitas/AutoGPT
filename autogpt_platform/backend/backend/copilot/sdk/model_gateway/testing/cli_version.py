import re
import subprocess
from pathlib import Path

_VERSION = re.compile(r"\b(\d+\.\d+\.\d+)\b")
_USER_AGENT = re.compile(r"(?:^|\s)claude-cli/(\d+\.\d+\.\d+)(?:\s|$)")


def binary_version(cli: Path) -> str:
    result = subprocess.run(
        [str(cli), "--version"], capture_output=True, text=True, check=True, timeout=15
    )
    match = _VERSION.search(result.stdout)
    assert match is not None, "The CLI binary did not report a version"
    return match.group(1)


def assert_cli_version(user_agents: list[str], expected: str) -> None:
    assert user_agents, "No CLI User-Agent was observed at the fake-backend gateway"
    for user_agent in user_agents:
        match = _USER_AGENT.search(user_agent)
        assert (
            match is not None
        ), f"Missing claude-cli version in User-Agent: {user_agent!r}"
        assert (
            match.group(1) == expected
        ), f"CLI User-Agent {user_agent!r} differs from binary version {expected}"
