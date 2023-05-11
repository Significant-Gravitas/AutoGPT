import os

import openai
import pytest

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file
from tests.integration.agent_utils import run_interaction_loop
from tests.utils import requires_api_key


@requires_api_key("OPENAI_API_KEY")
@pytest.mark.vcr
def test_write_file(writer_agent: Agent, patched_api_requestor) -> None:
    file_path = str(writer_agent.workspace.get_path("hello_world.txt"))
    try:
        run_interaction_loop(writer_agent, 200)
    # catch system exit exceptions
    except SystemExit:  # the agent returns an exception when it shuts down
        content = read_file(file_path)
        assert content == "Hello World", f"Expected 'Hello World', got {content}"
