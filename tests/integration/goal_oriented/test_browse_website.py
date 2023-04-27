<<<<<<< HEAD
import pytest

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file
from tests.integration.agent_utils import run_interaction_loop
from tests.utils import requires_api_key
=======
import os

import pytest

from autogpt.commands.file_operations import read_file
from autogpt.config import Config
from tests.integration.agent_factory import create_browser_agent
from tests.integration.agent_utils import run_interaction_loop
from tests.utils import get_workspace_file_path, requires_api_key

CFG = Config()
>>>>>>> 5d2360d (Refactor test browse website)


@requires_api_key("OPENAI_API_KEY")
@pytest.mark.vcr
<<<<<<< HEAD
def test_browse_website(browser_agent: Agent) -> None:
    file_path = browser_agent.workspace.get_path("browse_website.txt")
    try:
        run_interaction_loop(browser_agent, 40)
    # catch system exit exceptions
    except SystemExit:  # the agent returns an exception when it shuts down
        content = read_file(file_path)
=======
def test_browse_website(workspace) -> None:
    CFG.workspace_path = workspace.root
    CFG.file_logger_path = os.path.join(workspace.root, "file_logger.txt")

    file_name = get_workspace_file_path(workspace, "browse_website.txt")
    agent = create_browser_agent(workspace)
    try:
        run_interaction_loop(agent, 40)
    # catch system exit exceptions
    except SystemExit:  # the agent returns an exception when it shuts down
        content = read_file(file_name)
>>>>>>> 5d2360d (Refactor test browse website)
        assert "£25.89" in content, f"Expected £25.89, got {content}"
