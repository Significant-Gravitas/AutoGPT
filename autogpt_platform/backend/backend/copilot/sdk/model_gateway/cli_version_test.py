import pytest

from backend.copilot.sdk.model_gateway.testing.cli_version import assert_cli_version


@pytest.mark.parametrize(
    "agents",
    [
        [],
        [""],
        ["python-httpx/0.28.1"],
        ["claude-cli/2.1.296 (external, sdk-py)"],
        ["claude-cli/2.1.284-dev"],
    ],
)
def test_cli_version_guard_rejects_missing_or_different_versions(agents):
    with pytest.raises(AssertionError):
        assert_cli_version(agents, "2.1.284")


def test_cli_version_guard_checks_every_request():
    with pytest.raises(AssertionError):
        assert_cli_version(["claude-cli/2.1.284", "claude-cli/2.1.296"], "2.1.284")


def test_cli_version_guard_accepts_the_version_with_sdk_metadata():
    assert_cli_version(
        ["claude-cli/2.1.284 (external, sdk-py, agent-sdk/0.2.161)"], "2.1.284"
    )
