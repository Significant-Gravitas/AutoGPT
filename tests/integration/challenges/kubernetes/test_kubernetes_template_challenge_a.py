import pytest
import yaml
from pytest_mock import MockerFixture

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file
from autogpt.config import Config
from tests.integration.challenges.challenge_decorator.challenge_decorator import (
    challenge,
)
from tests.integration.challenges.utils import run_interaction_loop
from tests.utils import requires_api_key

CYCLE_COUNT = 3


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
@challenge
def test_kubernetes_template_challenge_a(
    kubernetes_agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
    patched_api_requestor: MockerFixture,
    config: Config,
    level_to_run: int,
) -> None:
    """
    Test the challenge_a function in a given agent by mocking user inputs
    and checking the output file content.

    Args:
        kubernetes_agent (Agent)
        monkeypatch (pytest.MonkeyPatch)
        config (Config)
        level_to_run (int)
    """
    run_interaction_loop(monkeypatch, kubernetes_agent, CYCLE_COUNT)

    file_path = str(kubernetes_agent.workspace.get_path("kube.yaml"))
    content = read_file(file_path, config)

    for word in ["apiVersion", "kind", "metadata", "spec"]:
        assert word in content, f"Expected the file to contain {word}"

    content = yaml.safe_load(content)
    for word in ["Service", "Deployment", "Pod"]:
        assert word in content["kind"], f"Expected the file to contain {word}"
