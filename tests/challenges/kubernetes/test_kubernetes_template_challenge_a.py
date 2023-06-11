import pytest
import yaml
from pytest_mock import MockerFixture

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file
from tests.challenges.challenge_decorator.challenge_decorator import challenge
from tests.challenges.utils import get_workspace_path, run_interaction_loop
from tests.utils import requires_api_key

CYCLE_COUNT = 3
OUTPUT_LOCATION = "kube.yaml"


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
@challenge
def test_kubernetes_template_challenge_a(
    kubernetes_agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
    patched_api_requestor: MockerFixture,
    level_to_run: int,
) -> None:
    """
    Test the challenge_a function in a given agent by mocking user inputs
    and checking the output file content.

    Args:
        kubernetes_agent (Agent)
        monkeypatch (pytest.MonkeyPatch)
        level_to_run (int)
    """
    run_interaction_loop(monkeypatch, kubernetes_agent, CYCLE_COUNT)

    file_path = get_workspace_path(kubernetes_agent, OUTPUT_LOCATION)
    content = read_file(file_path, kubernetes_agent)

    for word in ["apiVersion", "kind", "metadata", "spec"]:
        assert word in content, f"Expected the file to contain {word}"

    content = yaml.safe_load(content)
    for word in ["Service", "Deployment", "Pod"]:
        assert word in content["kind"], f"Expected the file to contain {word}"
