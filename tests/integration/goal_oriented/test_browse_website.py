import pytest

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file
from tests.integration.agent_utils import run_interaction_loop
from tests.utils import requires_api_key


@requires_api_key("OPENAI_API_KEY")
@pytest.mark.vcr
def test_browse_website(browser_agent: Agent, patched_api_requestor) -> None:
    file_path = browser_agent.workspace.get_path("browse_website.txt")
    try:
        run_interaction_loop(browser_agent, 120)
    # catch system exit exceptions
    except SystemExit:  # the agent returns an exception when it shuts down
        content = read_file(file_path)
        assert "£25.89" in content, f"Expected £25.89, got {content}"
