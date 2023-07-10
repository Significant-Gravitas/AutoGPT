import pytest
from pytest_mock import MockerFixture
from unittest.mock import patch, MagicMock
from autogpt.agent.agent import Agent
from autogpt.commands.web_selenium import browse_website, scrape_tags_with_selenium, scrape_images_with_selenium


@pytest.mark.vcr
@pytest.mark.requires_openai_api_key
def test_browse_website(agent: Agent, patched_api_requestor: MockerFixture):
    url = "https://barrel-roll.com"
    question = "How to execute a barrel roll"

    response = browse_website(url, question, agent)
    assert "Error" in response
    # Sanity check that the response is not too long
    assert len(response) < 200

@pytest.fixture
def mock_webdriver():
    mock = MagicMock()
    mock.execute_script.return_value = "<body></body>"
    return mock

@pytest.fixture
def mock_agent():
    return MagicMock()

@pytest.fixture
def mock_extraction_func():
    return MagicMock(return_value=MagicMock())

@pytest.fixture
def mock_formatting_func():
    return MagicMock(return_value=MagicMock())

def test_scrape_tags_with_selenium(mock_agent, 
                                       mock_extraction_func, mock_formatting_func):
    url = "http://test.com"
    result = scrape_tags_with_selenium(url, mock_agent, mock_extraction_func,
                                           mock_formatting_func)
    
    mock_webdriver.get.assert_called_once_with(url)
    mock_webdriver.execute_script.assert_called_once_with("return document.body.outerHTML;")
    mock_extraction_func.assert_called_once()
    mock_formatting_func.assert_called_once()

    assert result == mock_formatting_func.return_value
    
@pytest.mark.vcr
@pytest.mark.requires_openai_api_key 
def test_scrape_images_with_selenium(agent: Agent, patched_api_requestor: MockerFixture):
    url = "https://barrel-roll.com"
    question = "How to execute a barrel roll"

    response = scrape_images_with_selenium(url, agent)
    assert "Error" in response
    # Sanity check that the response is not too long
    assert len(response) < 200