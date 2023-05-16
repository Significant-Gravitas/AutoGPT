from selenium.common.exceptions import WebDriverException

from autogpt.commands.web_selenium import browse_website


def test_browse_website(config, mocker):
    mock = mocker.patch("autogpt.commands.web_selenium.scrape_text_with_selenium")
    mock.side_effect = WebDriverException("Some error happened")
    url = "https://google.com"
    question = "How to execute a barrel roll"

    response = browse_website(url, question)
    assert "Error" in response
    # Sanity check that the response is not too long
    assert len(response) < 200
