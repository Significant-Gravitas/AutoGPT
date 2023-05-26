"""Browse a webpage and summarize it using the LLM model"""
from __future__ import annotations

import requests
from bs4 import BeautifulSoup
from requests import Response

from autogpt.config import Config
from autogpt.processing.html import extract_hyperlinks, format_hyperlinks
from autogpt.url_utils.validators import validate_url

session = requests.Session()


@validate_url
def get_response(
    url: str, config: Config, timeout: int = 10
) -> tuple[None, str] | tuple[Response, None]:
    """Get the response from a URL

    Args:
        url (str): The URL to get the response from
        timeout (int): The timeout for the HTTP request

    Returns:
        tuple[None, str] | tuple[Response, None]: The response and error message

    Raises:
        ValueError: If the URL is invalid
        requests.exceptions.RequestException: If the HTTP request fails
    """
    try:
        session.headers.update({"User-Agent": config.user_agent})
        response = session.get(url, timeout=timeout)

        # Check if the response contains an HTTP error
        if response.status_code >= 400:
            return None, f"Error: HTTP {str(response.status_code)} error"

        return response, None
    except ValueError as ve:
        # Handle invalid URL format
        return None, f"Error: {str(ve)}"

    except requests.exceptions.RequestException as re:
        # Handle exceptions related to the HTTP request
        #  (e.g., connection errors, timeouts, etc.)
        return None, f"Error: {str(re)}"


def scrape_text(url: str, config: Config) -> str:
    """Scrape text from a webpage

    Args:
        url (str): The URL to scrape text from

    Returns:
        str: The scraped text
    """
    response, error_message = get_response(url, config)
    if error_message:
        return error_message
    if not response:
        return "Error: Could not get response"

    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)

    return text


def scrape_links(url: str, config: Config) -> str | list[str]:
    """Scrape links from a webpage

    Args:
        url (str): The URL to scrape links from

    Returns:
       str | list[str]: The scraped links
    """
    response, error_message = get_response(url, config)
    if error_message:
        return error_message
    if not response:
        return "Error: Could not get response"
    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    hyperlinks = extract_hyperlinks(soup, url)

    return format_hyperlinks(hyperlinks)
