import pytest

from backend.util.request import validate_url


def test_validate_url():
    with pytest.raises(ValueError):
        validate_url("localhost", [])

    with pytest.raises(ValueError):
        validate_url("192.168.1.1", [])

    with pytest.raises(ValueError):
        validate_url("127.0.0.1", [])

    with pytest.raises(ValueError):
        validate_url("0.0.0.0", [])

    validate_url("google.com", [])
    validate_url("github.com", [])
    validate_url("http://github.com", [])
