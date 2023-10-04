import pytest

from autogpt.workspace import Workspace
from tests.challenges.challenge_decorator.challenge_decorator import challenge
from tests.challenges.utils import run_challenge

CYCLE_COUNT = 2
USER_INPUTS = [
    "Use the browse_website command to visit http://books.toscrape.com/catalogue/meditations_33/index.html and answer the question 'What is the price of the book?'\nWrite the price of the book to a file named 'browse_website.txt'.'\nUse the task_complete command to complete the task.\nDo not use any other commands."
]


@challenge()
def test_browse_website(
    patched_api_requestor: None,
    monkeypatch: pytest.MonkeyPatch,
    level_to_run: int,
    challenge_name: str,
    workspace: Workspace,
    patched_make_workspace: pytest.fixture,
) -> None:
    run_challenge(
        challenge_name,
        level_to_run,
        monkeypatch,
        USER_INPUTS[level_to_run - 1],
        CYCLE_COUNT,
    )

    file_path = workspace.get_path("browse_website.txt")

    with open(file_path, "r") as file:
        content = file.read()
    assert "£25.89" in content, f"Expected £25.89, got {content}"
