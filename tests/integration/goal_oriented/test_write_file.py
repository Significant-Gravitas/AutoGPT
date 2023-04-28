import pytest

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file
from tests.utils import requires_api_key


@requires_api_key("OPENAI_API_KEY")
@pytest.mark.asyncio
@pytest.mark.timeout(40)
@pytest.mark.vcr
async def test_write_file(writer_agent: Agent) -> None:
    file_path = str(writer_agent.workspace.get_path("hello_world.txt"))
    try:
        await writer_agent.start_interaction_loop()
    # catch system exit exceptions
    except SystemExit:  # the agent returns an exception when it shuts down
        content = read_file(file_path)
        assert content == "Hello World", f"Expected 'Hello World', got {content}"
