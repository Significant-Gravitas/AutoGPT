from typing import List

import pytest

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file
from tests.challenges.challenge_decorator.challenge_decorator import challenge
from tests.challenges.utils import get_workspace_path, run_interaction_loop
from tests.utils import requires_api_key

CYCLE_COUNT_PER_LEVEL = [1, 1]
EXPECTED_OUTPUTS_PER_LEVEL = [
    {"hello_world.txt": ["Hello World"]},
    {"hello_world_1.txt": ["Hello World"], "hello_world_2.txt": ["Hello World"]},
]


@requires_api_key("OPENAI_API_KEY")
@pytest.mark.vcr
@challenge
def test_write_file(
    file_system_agents: List[Agent],
    patched_api_requestor: None,
    monkeypatch: pytest.MonkeyPatch,
    level_to_run: int,
) -> None:
    file_system_agent = file_system_agents[level_to_run - 1]
    run_interaction_loop(
        monkeypatch, file_system_agent, CYCLE_COUNT_PER_LEVEL[level_to_run - 1]
    )

    expected_outputs = EXPECTED_OUTPUTS_PER_LEVEL[level_to_run - 1]

    for file_name, expected_lines in expected_outputs.items():
        file_path = get_workspace_path(file_system_agent, file_name)
        content = read_file(file_path, file_system_agent)
        for expected_line in expected_lines:
            assert (
                expected_line in content
            ), f"Expected '{expected_line}' in file {file_name}, but it was not found"
