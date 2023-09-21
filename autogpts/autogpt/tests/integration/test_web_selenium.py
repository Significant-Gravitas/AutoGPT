import pytest

from autogpt.agents.agent import Agent
from autogpt.commands.web_selenium import BrowsingError, read_webpage


@pytest.mark.vcr
@pytest.mark.requires_openai_api_key
async def test_browse_website_nonexistent_url(
    agent: Agent, patched_api_requestor: None
):
    url = "https://auto-gpt-thinks-this-website-does-not-exist.com"
    question = "How to execute a barrel roll"

    with pytest.raises(BrowsingError, match=r"NAME_NOT_RESOLVED") as raised:
        await read_webpage(url=url, question=question, agent=agent)

        # Sanity check that the response is not too long
        assert len(raised.exconly()) < 200
