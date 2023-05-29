import pytest

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file
from autogpt.config import Config
from tests.integration.challenges.utils import run_interaction_loop
from tests.utils import requires_api_key

CYCLE_COUNT = 3


@requires_api_key("OPENAI_API_KEY")
@pytest.mark.vcr
def test_write_file(
    writer_agent: Agent,
    patched_api_requestor: None,
    monkeypatch: pytest.MonkeyPatch,
    config: Config,
) -> None:
    file_path = str(writer_agent.workspace.get_path("hello_world.txt"))
    run_interaction_loop(monkeypatch, writer_agent, CYCLE_COUNT)

    content = read_file(file_path, config)
    assert content == "Hello World", f"Expected 'Hello World', got {content}"
