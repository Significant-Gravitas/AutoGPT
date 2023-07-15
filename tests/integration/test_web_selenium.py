from unittest.mock import ANY, Mock, patch

import pytest
from pytest_mock import MockerFixture
from selenium.common.exceptions import WebDriverException

from autogpt.agents.agent import Agent
from autogpt.commands.web_selenium import browse_website, scrape_tag_links


@pytest.mark.vcr
@pytest.mark.requires_openai_api_key
def test_browse_website(agent: Agent, patched_api_requestor: MockerFixture):
    url = "https://barrel-roll.com"
    question = "How to execute a barrel roll"

    response = browse_website(url, question, agent)
    assert "error" in response.lower()
    # Sanity check that the response is not too long
    assert len(response) < 200


def test_scrape_tag_links_success():
    url = "http://testwebsite.com"
    tag_type = "a"
    agent = Mock(spec=Agent)
    include_keywords = ["test"]
    exclude_keywords = ["exclude"]

    with patch("autogpt.commands.web_selenium.get_webdriver") as mock_get_webdriver:
        with patch("autogpt.commands.web_selenium.load_url") as mock_load_url:
            with patch(
                "autogpt.commands.web_selenium.extract_tag_links"
            ) as mock_extract_tag_links:
                with patch(
                    "autogpt.commands.web_selenium.format_links"
                ) as mock_format_links:
                    mock_driver = Mock()
                    mock_driver.execute_script.return_value = (
                        "<html><head></head><body></body></html>"
                    )
                    mock_get_webdriver.return_value = mock_driver
                    mock_extract_tag_links.return_value = Mock()
                    mock_format_links.return_value = ["http://testwebsite.com/testlink"]

                    result = scrape_tag_links(
                        url, tag_type, agent, include_keywords, exclude_keywords
                    )

    assert result == ["http://testwebsite.com/testlink"]
    mock_get_webdriver.assert_called_once_with(agent)
    mock_load_url.assert_called_once_with(mock_driver, url)
    mock_extract_tag_links.assert_called_once_with(
        ANY, url, tag_type, include_keywords, exclude_keywords
    )
    mock_format_links.assert_called_once()


def test_scrape_tag_links_exception():
    url = "http://testwebsite.com"
    tag_type = "a"
    agent = Mock(spec=Agent)
    include_keywords = ["test"]
    exclude_keywords = ["exclude"]

    with patch("autogpt.commands.web_selenium.get_webdriver") as mock_get_webdriver:
        mock_get_webdriver.side_effect = WebDriverException

        with pytest.raises(WebDriverException):
            result = scrape_tag_links(
                url, tag_type, agent, include_keywords, exclude_keywords
            )

        mock_get_webdriver.assert_called_once_with(agent)
