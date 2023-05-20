import pytest
from pytest import raises

from autogpt.url_utils.validators import validate_url

"""
Code Analysis

Objective:
The objective of the 'validate_url' function is to validate URLs for any command that requires a URL as an argument. It checks if the URL is valid using a basic check, urllib check, and local file check. If the URL fails any of the validation tests, it raises a ValueError.

Inputs:
- func: A callable function that takes in any number of arguments and returns any type of output.

Flow:
- The 'validate_url' function takes in a callable function as an argument.
- It defines a wrapper function that takes in a URL and any number of arguments and keyword arguments.
- The wrapper function first checks if the URL starts with "http://" or "https://". If not, it raises a ValueError with the message "Invalid URL format".
- It then checks if the URL is valid using the 'is_valid_url' function. If not, it raises a ValueError with the message "Missing Scheme or Network location".
- It then checks if the URL is a local file using the 'check_local_file_access' function. If it is, it raises a ValueError with the message "Access to local files is restricted".
- If the URL passes all the validation tests, it sanitizes the URL using the 'sanitize_url' function and calls the original function with the sanitized URL and any other arguments and keyword arguments.
- The wrapper function returns the result of the original function.

Outputs:
- The 'validate_url' function returns the wrapper function that takes in a URL and any number of arguments and keyword arguments and returns the result of the original function.

Additional aspects:
- The 'validate_url' function uses the 'functools.wraps' decorator to preserve the original function's metadata, such as its name, docstring, and annotations.
- The 'validate_url' function uses the 'urlparse' function from the 'urllib.parse' module to parse the URL and extract its components.
- The 'validate_url' function uses the 'urljoin' function from the 'requests.compat' module to join the sanitized URL components back into a URL string.
"""


@validate_url
def dummy_method(url):
    return url


successful_test_data = (
    ("https://google.com/search?query=abc"),
    ("https://google.com/search?query=abc&p=123"),
    ("http://google.com/"),
    ("http://a.lot.of.domain.net/param1/param2"),
)


@pytest.mark.parametrize("url", successful_test_data)
def test_url_validation_succeeds(url):
    assert dummy_method(url) == url


bad_protocol_data = (
    ("htt://example.com"),
    ("httppp://example.com"),
    (" https://example.com"),
)


@pytest.mark.parametrize("url", bad_protocol_data)
def test_url_validation_fails_bad_protocol(url):
    with raises(ValueError, match="Invalid URL format"):
        dummy_method(url)


missing_loc = (("http://?query=q"),)


@pytest.mark.parametrize("url", missing_loc)
def test_url_validation_fails_bad_protocol(url):
    with raises(ValueError, match="Missing Scheme or Network location"):
        dummy_method(url)


local_file = (
    ("http://localhost"),
    ("https://localhost/"),
    ("http://2130706433"),
    ("https://2130706433"),
    ("http://127.0.0.1/"),
)


@pytest.mark.parametrize("url", local_file)
def test_url_validation_fails_local_path(url):
    with raises(ValueError, match="Access to local files is restricted"):
        dummy_method(url)


class TestValidateUrl:
    # Tests that the function successfully validates a valid URL with http:// or https:// prefix.
    def test_happy_path_valid_url(self):
        """Test that the function successfully validates a valid URL with http:// or https:// prefix"""

        @validate_url
        def test_func(url):
            return url

        assert test_func("https://www.google.com") == "https://www.google.com"
        assert test_func("http://www.google.com") == "http://www.google.com"

    # Tests that the function successfully validates a valid URL with additional path, parameters, and query string.
    def test_general_behavior_additional_path_parameters_query_string(self):
        """Test that the function successfully validates a valid URL with additional path, parameters, and query string"""

        @validate_url
        def test_func(url):
            return url

        assert (
            test_func("https://www.google.com/search?q=python")
            == "https://www.google.com/search?q=python"
        )

    # Tests that the function raises a ValueError if the URL is missing scheme or network location.
    def test_edge_case_missing_scheme_or_network_location(self):
        """Test that the function raises a ValueError if the URL is missing scheme or network location"""

        @validate_url
        def test_func(url):
            return url

        with pytest.raises(ValueError):
            test_func("www.google.com")

    # Tests that the function raises a ValueError if the URL has local file access.
    def test_edge_case_local_file_access(self):
        """Test that the function raises a ValueError if the URL has local file access"""

        @validate_url
        def test_func(url):
            return url

        with pytest.raises(ValueError):
            test_func("file:///etc/passwd")

    # Tests that the function sanitizes the URL by removing any unnecessary components.
    def test_general_behavior_sanitizes_url(self):
        """Test that the function sanitizes the URL by removing any unnecessary components"""

        @validate_url
        def test_func(url):
            return url

        assert (
            test_func("https://www.google.com/search?q=python#top")
            == "https://www.google.com/search?q=python"
        )

    # Tests that the function raises a ValueError if the URL has an invalid format (e.g. missing slashes).
    def test_general_behavior_invalid_url_format(self):
        """Test that the function raises a ValueError if the URL has an invalid format (e.g. missing slashes)"""

        @validate_url
        def test_func(url):
            return url

        with pytest.raises(ValueError):
            test_func("https:www.google.com")
