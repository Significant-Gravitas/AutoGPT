from typing import Any, Dict

import pytest
import yaml
from pytest_mock import MockerFixture

from autogpt.workspace import Workspace
from tests.challenges.challenge_decorator.challenge_decorator import challenge
from tests.challenges.utils import get_workspace_path, run_challenge

CYCLE_COUNT = 3
OUTPUT_LOCATION = "kube.yaml"
USER_INPUTS = ["Write a simple kubernetes deployment file and save it as a kube.yaml."]


@challenge()
def test_kubernetes_template_challenge_a(
    monkeypatch: pytest.MonkeyPatch,
    patched_api_requestor: MockerFixture,
    level_to_run: int,
    challenge_name: str,
    workspace: Workspace,
    patched_make_workspace: pytest.fixture,
) -> None:
    """
    Test the challenge_a function in a given agent by mocking user inputs
    and checking the output file content.

    Args:
        kubernetes_agent (Agent)
        monkeypatch (pytest.MonkeyPatch)
        level_to_run (int)
    """
    run_challenge(
        challenge_name,
        level_to_run,
        monkeypatch,
        USER_INPUTS[level_to_run - 1],
        CYCLE_COUNT,
    )

    file_path = get_workspace_path(workspace, OUTPUT_LOCATION)
    with open(file_path, "r") as file:
        content_string = file.read()

    for word in ["apiVersion", "kind", "metadata", "spec"]:
        assert word in content_string, f"Expected the file to contain {word}"

    yaml_as_dict: Dict[str, Any] = yaml.safe_load(content_string)
    for word in ["Service", "Deployment", "Pod"]:
        assert word in yaml_as_dict.get(
            "kind", ""
        ), f"Expected the file to contain {word}"
