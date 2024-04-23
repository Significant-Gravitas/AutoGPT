import functools
import re
from inspect import signature
from typing import Callable, ParamSpec, TypeVar
from urllib.parse import urljoin, urlparse

P = ParamSpec("P")
T = TypeVar("T")


def validate_url(func: Callable[P, T]) -> Callable[P, T]:
    """
    The method decorator validate_url is used to validate urls for any command that
    requires a url as an argument.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sig = signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        url = bound_args.arguments.get("url")
        if url is None:
            raise ValueError("URL is required for this function")

        if not re.match(r"^https?://", url):
            raise ValueError(
                "Invalid URL format: URL must start with http:// or https://"
            )
        if not is_valid_url(url):
            raise ValueError("Missing Scheme or Network location")
        if check_local_file_access(url):
            raise ValueError("Access to local files is restricted")
        if len(url) > 2000:
            raise ValueError("URL is too long")

        bound_args.arguments["url"] = sanitize_url(url)

        return func(*bound_args.args, **bound_args.kwargs)

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
