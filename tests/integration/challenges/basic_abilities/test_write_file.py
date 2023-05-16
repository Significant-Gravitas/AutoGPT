import pytest
from pytest_mock import MockerFixture

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file
from tests.integration.challenges.utils import run_interaction_loop
from tests.utils import requires_api_key

CYCLE_COUNT = 3


@requires_api_key("OPENAI_API_KEY")
@pytest.mark.vcr
def test_write_file(
    writer_agent: Agent,
    patched_api_requestor: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = str(writer_agent.workspace.get_path("hello_world.txt"))
    run_interaction_loop(monkeypatch, writer_agent, CYCLE_COUNT)

    content = read_file(file_path)
    assert content == "Hello World", f"Expected 'Hello World', got {content}"
