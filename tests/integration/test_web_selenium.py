import pytest
from pytest_mock import MockerFixture

from autogpt.agent.agent import Agent
from autogpt.commands.web_selenium import browse_website


@pytest.mark.vcr
@pytest.mark.requires_openai_api_key
def test_browse_website(agent: Agent, patched_api_requestor: MockerFixture):
    url = "https://barrel-roll.com"
    question = "How to execute a barrel roll"

    response = browse_website(url, question, agent)
    assert "Error" in response
    # Sanity check that the response is not too long
    assert len(response) < 200
