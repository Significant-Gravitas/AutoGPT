import pytest

from autogpt.agent import Agent
from tests.integration.challenges.utils import run_interaction_loop
from tests.utils import requires_api_key

CYCLE_COUNT = 2


@requires_api_key("OPENAI_API_KEY")
@pytest.mark.vcr
def test_browse_website(
    browser_agent: Agent,
    patched_api_requestor: None,
    monkeypatch: pytest.MonkeyPatch,
    # config: Config,
) -> None:
    file_path = browser_agent.workspace.get_path("browse_website.txt")
    run_interaction_loop(monkeypatch, browser_agent, CYCLE_COUNT)

    # content = read_file(file_path, config)
    content = open(file_path, encoding="utf-8").read()
    assert "£25.89" in content, f"Expected £25.89, got {content}"
