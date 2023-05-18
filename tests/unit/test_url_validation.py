import pytest
from pytest import raises

from autogpt.url_utils.validators import validate_url


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
