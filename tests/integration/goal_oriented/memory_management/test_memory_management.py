import os

import pytest

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file, write_to_file
from autogpt.config import Config
from tests.integration.agent_utils import run_interaction_loop
from tests.utils import get_workspace_file_path, requires_api_key

CFG = Config()

instructions = """Your task_id is 2314
Read the file file_1.txt"""

file_1 = """Go to file_2.txt"""

file_2 = """Go to file_3.txt"""
file_3 = """Go to file_4.txt"""
file_4 = """Go to file_5.txt"""
file_5 = """Go to file_6.txt"""
file_6 = """Go to file_7.txt"""
file_7 = """Go to file_8.txt"""
file_8 = """Go to file_9.txt"""
file_9 = """Go to file_10.txt"""
file_10 = """Go to file_11.txt"""
file_11 = """Go to file_12.txt"""
file_12 = """Go to file_13.txt"""
file_13 = """Go to file_14.txt"""
file_14 = """Go to file_15.txt"""
file_15 = """Go to file_16.txt"""
file_16 = """Go to file_17.txt"""
file_17 = """Go to file_18.txt"""
file_18 = """Go to file_19.txt"""
file_19 = """Go to file_to_read.txt"""

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
    
    file_1_path = str(memory_management_agent.workspace.get_path("file_1.txt"))
    file_2_path = str(memory_management_agent.workspace.get_path("file_2.txt"))
    file_3_path = str(memory_management_agent.workspace.get_path("file_3.txt"))
    file_4_path = str(memory_management_agent.workspace.get_path("file_4.txt"))
    file_5_path = str(memory_management_agent.workspace.get_path("file_5.txt"))
    file_6_path = str(memory_management_agent.workspace.get_path("file_6.txt"))
    file_7_path = str(memory_management_agent.workspace.get_path("file_7.txt"))
    file_8_path = str(memory_management_agent.workspace.get_path("file_8.txt"))
    file_9_path = str(memory_management_agent.workspace.get_path("file_9.txt"))
    file_10_path = str(memory_management_agent.workspace.get_path("file_10.txt"))
    file_11_path = str(memory_management_agent.workspace.get_path("file_11.txt"))
    file_12_path = str(memory_management_agent.workspace.get_path("file_12.txt"))
    file_13_path = str(memory_management_agent.workspace.get_path("file_13.txt"))
    file_14_path = str(memory_management_agent.workspace.get_path("file_14.txt"))
    file_15_path = str(memory_management_agent.workspace.get_path("file_15.txt"))
    file_16_path = str(memory_management_agent.workspace.get_path("file_16.txt"))
    file_17_path = str(memory_management_agent.workspace.get_path("file_17.txt"))
    file_18_path = str(memory_management_agent.workspace.get_path("file_18.txt"))
    file_19_path = str(memory_management_agent.workspace.get_path("file_19.txt"))
    
    write_to_file(file_1_path, file_1)
    write_to_file(file_2_path, file_2)
    write_to_file(file_3_path, file_3)
    write_to_file(file_4_path, file_4)
    write_to_file(file_5_path, file_5)
    write_to_file(file_6_path, file_6)
    write_to_file(file_7_path, file_7)
    write_to_file(file_8_path, file_8)
    write_to_file(file_9_path, file_9)
    write_to_file(file_10_path, file_10)
    write_to_file(file_11_path, file_11)
    write_to_file(file_12_path, file_12)
    write_to_file(file_13_path, file_13)
    write_to_file(file_14_path, file_14)
    write_to_file(file_15_path, file_15)
    write_to_file(file_16_path, file_16)
    write_to_file(file_17_path, file_17)
    write_to_file(file_18_path, file_18)
    write_to_file(file_19_path, file_19)


    try:
        run_interaction_loop(memory_management_agent, 40)
    # catch system exit exceptions
    except SystemExit:  # the agent returns an exception when it shuts down
        file_path = str(memory_management_agent.workspace.get_path("output.txt"))
        content = read_file(file_path)
        assert "2314" in content, f"Expected the file to contain 2314"
