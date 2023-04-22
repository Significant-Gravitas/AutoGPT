"""Browse a webpage and summarize it using the LLM model"""
from __future__ import annotations

import requests
import traceback
from bs4 import BeautifulSoup
from requests import Response
from requests.compat import urljoin
from typing import Dict, Any, Union, List

from autogpt.config import Config
from autogpt.processing.html import extract_hyperlinks, format_hyperlinks
from autogpt.url_utils.validators import validate_url
from autogpt.commands.command import command
from autogpt.logs import logger

CFG = Config()

session = requests.Session()
session.headers.update({"User-Agent": CFG.user_agent})

JSONType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

@command("make_http_request", "Make an HTTP request", '"url": "<url>", "method": "<method>", "auth_token": "<auth_token>", "data": "<data>"')
def make_http_request(url: str, method: str = "GET", auth_token: str = None, data: JSONType = None):
    if not is_valid_url(url):
        logger.error(f"Invalid URL: {url}")
        return "Error: Invalid URL"

    sanitized_url: str = sanitize_url(url)
    logger._log(f"Sending HTTP request: {method} {sanitized_url}")

    # Set up headers with API key
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    if (method == "POST" or method == "PUT"):
        headers["Content-Type"] = "application/json"

    # Send request
    try:
        response = requests.request(method, sanitized_url, json=data, headers=headers)
        logger._log(f"HTTP request sent successfully: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Error making HTTP request: {e}")
        logger.error(traceback.format_exc())
        return e

@validate_url
def get_response(
    url: str, timeout: int = 10
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


def scrape_text(url: str) -> str:
    """Scrape text from a webpage

    Args:
        url (str): The URL to scrape text from

    Returns:
        str: The scraped text
    """
    response, error_message = get_response(url)
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


def scrape_links(url: str) -> str | list[str]:
    """Scrape links from a webpage

    Args:
        url (str): The URL to scrape links from

    Returns:
       str | list[str]: The scraped links
    """
    response, error_message = get_response(url)
    if error_message:
        return error_message
    if not response:
        return "Error: Could not get response"
    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    hyperlinks = extract_hyperlinks(soup, url)

    return format_hyperlinks(hyperlinks)


def create_message(chunk, question):
    """Create a message for the user to summarize a chunk of text"""
    return {
        "role": "user",
        "content": f'"""{chunk}""" Using the above text, answer the following'
        f' question: "{question}" -- if the question cannot be answered using the'
        " text, summarize the text.",
    }
