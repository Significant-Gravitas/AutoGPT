import pytest

from autogpt.agents.agent import Agent
from autogpt.commands.web_selenium import BrowsingError, browse_website


@pytest.mark.vcr
@pytest.mark.requires_openai_api_key
def test_browse_website_nonexistent_url(agent: Agent, patched_api_requestor: None):
    url = "https://barrel-roll.com"
    question = "How to execute a barrel roll"

    with pytest.raises(BrowsingError, match=r"CONNECTION_CLOSED") as raised:
        browse_website(url, question, agent)

        # Sanity check that the response is not too long
        assert len(raised.exconly()) < 200
