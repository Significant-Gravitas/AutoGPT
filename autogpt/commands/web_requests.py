"""Browse a webpage and summarize it using the LLM model"""
from typing import List, Tuple, Union
from urllib.parse import urljoin, urlparse

import requests
from requests.compat import urljoin
from requests import Response
from bs4 import BeautifulSoup
from lxml import etree

from autogpt.config import Config
from autogpt.memory import get_memory
from autogpt.processing.html import extract_hyperlinks, format_hyperlinks

CFG = Config()
memory = get_memory(CFG)

session = requests.Session()
session.headers.update({"User-Agent": CFG.user_agent})


def is_valid_url(url: str) -> bool:
    """Check if the URL is valid

    Args:
        url (str): The URL to check

    Returns:
        bool: True if the URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def sanitize_url(url: str) -> str:
    """Sanitize the URL

    Args:
        url (str): The URL to sanitize

    Returns:
        str: The sanitized URL
    """
    return urljoin(url, urlparse(url).path)


def check_local_file_access(url: str) -> bool:
    """Check if the URL is a local file

    Args:
        url (str): The URL to check

    Returns:
        bool: True if the URL is a local file, False otherwise
    """
    local_prefixes = [
        "file:///",
        "file://localhost",
        "http://localhost",
        "https://localhost",
    ]
    return any(url.startswith(prefix) for prefix in local_prefixes)


def get_response(
    url: str, timeout: int = 10
) -> Union[Tuple[None, str], Tuple[Response, None]]:
    """Get the response from a URL

    Args:
        url (str): The URL to get the response from
        timeout (int): The timeout for the HTTP request

    Returns:
        Tuple[None, str] | Tuple[Response, None]: The response and error message

    Raises:
        ValueError: If the URL is invalid
        requests.exceptions.RequestException: If the HTTP request fails
    """
    try:
        # Restrict access to local files
        if check_local_file_access(url):
            raise ValueError("Access to local files is restricted")

        # Most basic check if the URL is valid:
        if not url.startswith("http://") and not url.startswith("https://"):
            raise ValueError("Invalid URL format")

        sanitized_url = sanitize_url(url)

        response = session.get(sanitized_url, timeout=timeout)

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


def scrape_links(url: str) -> Union[str, List[str]]:
    """Scrape links from a webpage

    Args:
        url (str): The URL to scrape links from

    Returns:
        Union[str, List[str]]: The scraped links
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

def get_xpath(element):
    """Gets the full xpath of an element

    Args:
        element (lxml.etree._Element): The element to get the full XPath for

    Returns:
        A str of the full xpath of the element
    """
    parts = []
    while element is not None:
        parent = element.getparent()
        if parent is None:
            break
        siblings = [sib for sib in parent if sib.tag == element.tag]
        if len(siblings) > 1:
            parts.insert(0, f"{element.tag}[{siblings.index(element) + 1}]")
        else:
            parts.insert(0, element.tag)
        element = parent
    return "/".join(parts)


def scrape_buttons(url: str):
    """Scrape buttons from a webpage
    
    Args:
        url (str): The URL to scrape links from

    Returns:
        A 2D list of [str: button text, full xpath]
    """

    response, error_message = get_response(url)
    if error_message:
        return error_message
    if not response:
        return "Error: Could not get response"
    soup = BeautifulSoup(response.text, "html.parser")

    buttons = soup.find_all("button")

    button_data = []
    for button in buttons:
        button_name = button.text.strip()
        button_element = etree.fromstring(str(button))
        button_xpath = get_xpath(button_element)
        button_data.append([button_name, button_xpath])

    return button_data

def scrape_input_text_fields(url: str):
    """Scrape all input text fields from a website

    Args:
        url (str): The URL of the website to scrape

    Returns:
        List[List[str]]: A 2D array where each inner array is [input field name or id, full xpath to the input field]
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "lxml")
    input_text_fields = soup.find_all("input", {"type": "text"})

    field_data = []
    for field in input_text_fields:
        field_name = field.get("name", field.get("id", "Unnamed"))
        field_element = etree.fromstring(str(field))
        field_xpath = get_xpath(field_element)
        field_data.append([field_name, field_xpath])

    return field_data

def scrape_select_input_fields(url: str):
    """Scrape all select input fields and their options from a website

    Args:
        url (str): The URL of the website to scrape

    Returns:
        List[List[str]]: A 2D array where each inner array is [select input field name or id, full xpath to the select input field, list of option values]
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "lxml")
    select_input_fields = soup.find_all("select")

    field_data = []
    for field in select_input_fields:
        field_name = field.get("name", field.get("id", "Unnamed"))
        field_element = etree.fromstring(str(field))
        field_xpath = get_xpath(field_element)

        options = []
        for option in field.find_all("option"):
            option_value = option.get("value")
            options.append(option_value)

        field_data.append([field_name, field_xpath, options])

    return field_data