import pytest
from pytest_mock import MockerFixture

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file
from tests.integration.challenges.utils import run_interaction_loop
from tests.utils import requires_api_key

CYCLE_COUNT = 2


@requires_api_key("OPENAI_API_KEY")
@pytest.mark.vcr
def test_browse_website(
    browser_agent: Agent,
    patched_api_requestor: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = browser_agent.workspace.get_path("browse_website.txt")
    run_interaction_loop(monkeypatch, browser_agent, CYCLE_COUNT)

    content = read_file(file_path)
    assert "£25.89" in content, f"Expected £25.89, got {content}"
