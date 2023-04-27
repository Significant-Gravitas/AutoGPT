import os

import pytest

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file, write_to_file
from autogpt.config import Config
from tests.integration.agent_utils import run_interaction_loop
from tests.utils import get_workspace_file_path, requires_api_key

CFG = Config()

instructions = """Your task_id is 2314
Read the file file_to_read.txt"""

file_to_read = """Write your task_id to the file output.txt.
Use the task_complete command"""


@requires_api_key("OPENAI_API_KEY")
@pytest.mark.vcr
# @pytest.skip # Remove when challenge completed
def test_memory_management(memory_management_agent: Agent) -> None:
    """DESCRIPTION OF THE TEST
        An agent that just starts a life doesn't have a long message history. So it shouldn't have memories.
        Because memories are only needed when
    """
    file_to_read_path = str(memory_management_agent.workspace.get_path("file_to_read.txt"))
    instructions_path = str(memory_management_agent.workspace.get_path("instructions.txt"))
    write_to_file(file_to_read_path, file_to_read)
    write_to_file(instructions_path, instructions)

    try:
        run_interaction_loop(memory_management_agent, 40)
    # catch system exit exceptions
    except SystemExit:  # the agent returns an exception when it shuts down
        file_path = str(memory_management_agent.workspace.get_path("output.txt"))
        content = read_file(file_path)
        assert "2314" in content, f"Expected the file to contain 2314"
