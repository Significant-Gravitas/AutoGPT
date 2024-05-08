import pytest
from forge.components.web_selenium import BrowsingError, WebSeleniumComponent

from autogpt.agents.agent import Agent


@pytest.fixture
def web_selenium_component(agent: Agent):
    return agent.web_selenium


@pytest.mark.vcr
@pytest.mark.requires_openai_api_key
@pytest.mark.asyncio
async def test_browse_website_nonexistent_url(
    web_selenium_component: WebSeleniumComponent, cached_openai_client: None
):
    url = "https://auto-gpt-thinks-this-website-does-not-exist.com"
    question = "How to execute a barrel roll"

    with pytest.raises(BrowsingError, match="NAME_NOT_RESOLVED") as raised:
        await web_selenium_component.read_webpage(url=url, question=question)

        # Sanity check that the response is not too long
        assert len(raised.exconly()) < 200
