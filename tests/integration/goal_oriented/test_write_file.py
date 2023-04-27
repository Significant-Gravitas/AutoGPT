import os

import pytest
import vcr

from autogpt.agent import Agent
from autogpt.commands.command import CommandRegistry
from autogpt.commands.file_operations import LOG_FILE, delete_file, read_file
from autogpt.config import Config, check_openai_api_key
from autogpt.projects.project_config_broker import ProjectConfigBroker
from autogpt.memory import get_memory
from tests.integration.agent_factory import create_writer_agent
from tests.integration.agent_utils import run_interaction_loop
from tests.utils import requires_api_key
from autogpt.prompts.prompt import construct_full_prompt

CFG = Config()

@pytest.mark.integration_test
@requires_api_key("OPENAI_API_KEY")
@pytest.mark.vcr
def test_write_file(workspace) -> None:
    CFG.workspace_path = workspace.root
    CFG.file_logger_path = os.path.join(workspace.root, "file_logger.txt")

    file_name = str(workspace.get_path("hello_world.txt"))
    agent = create_writer_agent(workspace)
    try:
        run_interaction_loop(agent, 40)
    # catch system exit exceptions
    except SystemExit:  # the agent returns an exception when it shuts down
        content = read_file(file_name)
        assert content == "Hello World", f"Expected 'Hello World', got {content}"
