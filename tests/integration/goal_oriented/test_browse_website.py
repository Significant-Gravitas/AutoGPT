import pytest

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file
from tests.utils import requires_api_key


@requires_api_key("OPENAI_API_KEY")
@pytest.mark.asyncio
@pytest.mark.timeout(40)
@pytest.mark.vcr
async def test_browse_website(browser_agent: Agent) -> None:
    file_path = browser_agent.workspace.get_path("browse_website.txt")
    try:
        await browser_agent.start_interaction_loop()
    # catch system exit exceptions
    except SystemExit:  # the agent returns an exception when it shuts down
        content = read_file(file_path)
        assert "£25.89" in content, f"Expected £25.89, got {content}"
