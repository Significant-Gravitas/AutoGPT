import os
from typing import Any, Dict
from unittest.mock import Mock, patch

import requests


def test_make_request_and_assert() -> None:
    result = make_request_and_assert()
    print(result)
    expected_result = "OK"
    error_message = f"AssertionError: Expected the output to be {expected_result}"
    print(error_message)
    assert result == expected_result, error_message


def make_assertion() -> None:
    if os.environ.get("MOCK_TEST", "False").lower() == "true":
        mock_response = Mock(requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "OK"}

        with patch("requests.get", return_value=mock_response):
            make_request_and_assert()
    else:
        make_request_and_assert()


def make_request_and_assert() -> Dict[str, Any]:
    response = requests.get("http://localhost:8079/health")
    if response.status_code != 200:
        raise AssertionError(
            f"Expected status code 200, but got {response.status_code}"
        )

    return response.json()


if __name__ == "__main__":
    # test for the case when server is healthy
    test_make_request_and_assert()
