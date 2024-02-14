import functools
import re
from typing import Any, Callable, ParamSpec, TypeVar
from urllib.parse import urljoin, urlparse
from autogpt.config import Config

P = ParamSpec("P")
T = TypeVar("T")
config = Config

def validate_url(func: Callable[P, T]) -> Callable[P, T]:
    """
    The method decorator validate_url is used to validate urls for any command that
    requires a url as an argument.
    """

    @functools.wraps(func)
    def wrapper(url: str, *args, **kwargs) -> Any:
        """Check if the URL is valid and not a local file accessor.

        Args:
            url (str): The URL to check

        Returns:
            the result of the wrapped function

        Raises:
            ValueError if the url fails any of the validation tests
        """
        web_policy = config.web_policy
        url_list = config.url_list
        
        # Most basic check if the URL is valid:
        if not re.match(r"^https?://", url):
            raise ValueError("Invalid URL format")
        if not is_valid_url(url):
            raise ValueError("Missing Scheme or Network location")
        # Restrict access to local files
        if check_local_file_access(url):
            raise ValueError("Access to local files is restricted")
        # Check URL length
        if len(url) > 2000:
            raise ValueError("URL is too long")
        if web_policy:
            if url not in url_list:
                raise ValueError("URL Not Whitelisted")
        elif url in url_list:
            raise ValueError("URL Blacklisted.")

        return func(sanitize_url(url), *args, **kwargs)

    return wrapper


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
    parsed_url = urlparse(url)
    reconstructed_url = f"{parsed_url.path}{parsed_url.params}?{parsed_url.query}"
    return urljoin(url, reconstructed_url)


def check_local_file_access(url: str) -> bool:
    """Check if the URL is a local file

    Args:
        url (str): The URL to check

    Returns:
        bool: True if the URL is a local file, False otherwise
    """
    # List of local file prefixes
    local_file_prefixes = [
        "file:///",
        "file://localhost",
    ]

    return any(url.startswith(prefix) for prefix in local_file_prefixes)
