import pytest
import yaml
from pytest_mock import MockerFixture

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file
from tests.challenges.challenge_decorator.challenge_decorator import challenge
from tests.challenges.utils import get_workspace_path_from_agent, run_interaction_loop

CYCLE_COUNT = 3
OUTPUT_LOCATION = "kube.yaml"


@challenge()
def test_kubernetes_template_challenge_a(
    kubernetes_agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
    patched_api_requestor: MockerFixture,
    level_to_run: int,
    challenge_name: str,
) -> None:
    """
    Test the challenge_a function in a given agent by mocking user inputs
    and checking the output file content.

    Args:
        kubernetes_agent (Agent)
        monkeypatch (pytest.MonkeyPatch)
        level_to_run (int)
    """
    run_interaction_loop(
        monkeypatch, kubernetes_agent, CYCLE_COUNT, challenge_name, level_to_run
    )

    file_path = get_workspace_path_from_agent(kubernetes_agent, OUTPUT_LOCATION)
    content = read_file(file_path, kubernetes_agent)

    for word in ["apiVersion", "kind", "metadata", "spec"]:
        assert word in content, f"Expected the file to contain {word}"

    content = yaml.safe_load(content)
    for word in ["Service", "Deployment", "Pod"]:
        assert word in content["kind"], f"Expected the file to contain {word}"
