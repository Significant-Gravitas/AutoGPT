import pytest

from autogpt.agent import Agent
from tests.challenges.challenge_decorator.challenge_decorator import challenge
from tests.challenges.utils import run_interaction_loop

CYCLE_COUNT = 2


@challenge()
def test_browse_website(
    browser_agent: Agent,
    patched_api_requestor: None,
    monkeypatch: pytest.MonkeyPatch,
    level_to_run: int,
) -> None:
    file_path = browser_agent.workspace.get_path("browse_website.txt")
    run_interaction_loop(monkeypatch, browser_agent, CYCLE_COUNT)

    # content = read_file(file_path, config)
    content = open(file_path, encoding="utf-8").read()
    assert "£25.89" in content, f"Expected £25.89, got {content}"
