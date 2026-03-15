import pytest
from pytest import raises

from .url_validator import validate_url


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


@pytest.mark.parametrize(
    "url,expected_error",
    [
        ("htt://example.com", "Invalid URL format"),
        ("httppp://example.com", "Invalid URL format"),
        (" https://example.com", "Invalid URL format"),
        ("http://?query=q", "Missing Scheme or Network location"),
    ],
)
def test_url_validation_fails_invalid_url(url, expected_error):
    with raises(ValueError, match=expected_error):
        dummy_method(url)


local_file = (
    ("file://localhost"),
    ("file://localhost/home/reinier/secrets.txt"),
    ("file:///home/reinier/secrets.txt"),
    ("file:///C:/Users/Reinier/secrets.txt"),
)


@pytest.mark.parametrize("url", local_file)
def test_url_validation_fails_local_path(url):
    with raises(ValueError):
        dummy_method(url)


def test_happy_path_valid_url():
    """
    Test that the function successfully validates a valid URL with `http://` or
    `https://` prefix.
    """

    @validate_url
    def test_func(url):
        return url

    assert test_func("https://www.google.com") == "https://www.google.com"
    assert test_func("http://www.google.com") == "http://www.google.com"


def test_general_behavior_additional_path_parameters_query_string():
    """
    Test that the function successfully validates a valid URL with additional path,
    parameters, and query string.
    """

    @validate_url
    def test_func(url):
        return url

    assert (
        test_func("https://www.google.com/search?q=python")
        == "https://www.google.com/search?q=python"
    )


def test_edge_case_missing_scheme_or_network_location():
    """
    Test that the function raises a ValueError if the URL is missing scheme or
    network location.
    """

    @validate_url
    def test_func(url):
        return url

    with pytest.raises(ValueError):
        test_func("www.google.com")


def test_edge_case_local_file_access():
    """Test that the function raises a ValueError if the URL has local file access"""

    @validate_url
    def test_func(url):
        return url

    with pytest.raises(ValueError):
        test_func("file:///etc/passwd")


def test_general_behavior_sanitizes_url():
    """Test that the function sanitizes the URL by removing unnecessary components"""

    @validate_url
    def test_func(url):
        return url

    assert (
        test_func("https://www.google.com/search?q=python#top")
        == "https://www.google.com/search?q=python"
    )


def test_general_behavior_invalid_url_format():
    """
    Test that the function raises a ValueError if the URL has an invalid format
    (e.g. missing slashes)
    """

    @validate_url
    def test_func(url):
        return url

    with pytest.raises(ValueError):
        test_func("https:www.google.com")


def test_url_with_special_chars():
    """
    Tests that the function can handle URLs that contain unusual but valid characters.
    """
    url = "https://example.com/path%20with%20spaces"
    assert dummy_method(url) == url


def test_extremely_long_url():
    """
    Tests that the function raises a ValueError if the URL is over 2000 characters.
    """
    url = "http://example.com/" + "a" * 2000
    with raises(ValueError, match="URL is too long"):
        dummy_method(url)


def test_internationalized_url():
    """
    Tests that the function can handle internationalized URLs with non-ASCII characters.
    """
    url = "http://例子.测试"
    assert dummy_method(url) == url
