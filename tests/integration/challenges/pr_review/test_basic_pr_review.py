from types import SimpleNamespace

import pytest

from autogpt.workspace import Workspace
from tests.integration.challenges.pr_review.base import run_tests
from tests.utils import requires_api_key

PR_LINK = "https://github.com/merwanehamadi/Auto-GPT/pull/116"
PARAMETERS = SimpleNamespace(
    source_branch_name="useless-comment",
    source_repo_user="merwanehamadi",

    # PR information
    title="Useless comment",
    body="Useless comment",
    # time allowed to run
    cycle_count=3,
    # PR success criteria
    approved=False,
    contains={"bad_variable_name.py": ["variable"]},
)


@requires_api_key("OPENAI_API_KEY")
def test_basic_pr_review(monkeypatch: pytest.MonkeyPatch, workspace: Workspace) -> None:
    run_tests(PARAMETERS, monkeypatch, workspace)
