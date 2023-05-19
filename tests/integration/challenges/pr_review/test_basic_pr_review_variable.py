from types import SimpleNamespace
import pytest
from autogpt.workspace import Workspace
from tests.integration.challenges.pr_review.base import run_tests
from tests.utils import requires_api_key
PR_LINK = "https://github.com/merwanehamadi/Auto-GPT/pull/116"
PARAMETERS = SimpleNamespace(
    cycle_count=6,
    pr_target_repo_user="merwanehamadi",
    pr_target_repo_name="Auto-GPT",
    source_branch_name="useless-comment",
    title="Useless comment",
    body="Useless comment",
    approved=True,
    contains = {'bad_variables.py': 'not used'}
)

@requires_api_key("OPENAI_API_KEY")
def test_basic_pr_review(
    monkeypatch: pytest.MonkeyPatch, workspace: Workspace
) -> None:
    run_tests(PARAMETERS, monkeypatch, workspace)
