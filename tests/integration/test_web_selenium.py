from pytest_mock import MockerFixture

from autogpt.commands.web_selenium import browse_website
from autogpt.config import Config


def test_browse_website(config: Config, patched_api_requestor: MockerFixture):
    url = "https://barrel-roll.com"
    question = "How to execute a barrel roll"

    response = browse_website(url, question, config)
    assert "Error" in response
    # Sanity check that the response is not too long
    assert len(response) < 200
