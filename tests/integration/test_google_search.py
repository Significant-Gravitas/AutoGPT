import json

import pytest
from googleapiclient.errors import HttpError

from autogpt.commands.google_search import (
    google_official_search,
    google_search,
    safe_google_results,
)


@pytest.mark.parametrize(
    "query, expected_output",
    [("test", "test"), (["test1", "test2"], '["test1", "test2"]')],
)
def test_safe_google_results(query, expected_output):
    result = safe_google_results(query)
    assert isinstance(result, str)
    assert result == expected_output


def test_safe_google_results_invalid_input():
    with pytest.raises(AttributeError):
        safe_google_results(123)


@pytest.mark.parametrize(
    "query, num_results, expected_output, return_value",
    [
        (
            "test",
            1,
            '[\n    {\n        "title": "Result 1",\n        "link": "https://example.com/result1"\n    }\n]',
            [{"title": "Result 1", "link": "https://example.com/result1"}],
        ),
        ("", 1, "[]", []),
        ("no results", 1, "[]", []),
    ],
)
def test_google_search(
    query, num_results, expected_output, return_value, mocker, config
):
    mock_ddg = mocker.Mock()
    mock_ddg.return_value = return_value

    mocker.patch("autogpt.commands.google_search.DDGS.text", mock_ddg)
    actual_output = google_search(query, config, num_results=num_results)
    expected_output = safe_google_results(expected_output)
    assert actual_output == expected_output


@pytest.fixture
def mock_googleapiclient(mocker):
    mock_build = mocker.patch("googleapiclient.discovery.build")
    mock_service = mocker.Mock()
    mock_build.return_value = mock_service
    return mock_service.cse().list().execute().get


@pytest.mark.parametrize(
    "query, num_results, search_results, expected_output",
    [
        (
            "test",
            3,
            [
                {"link": "http://example.com/result1"},
                {"link": "http://example.com/result2"},
                {"link": "http://example.com/result3"},
            ],
            [
                "http://example.com/result1",
                "http://example.com/result2",
                "http://example.com/result3",
            ],
        ),
        ("", 3, [], []),
    ],
)
def test_google_official_search(
    query, num_results, expected_output, search_results, mock_googleapiclient, config
):
    mock_googleapiclient.return_value = search_results
    actual_output = google_official_search(query, config, num_results=num_results)
    assert actual_output == safe_google_results(expected_output)


@pytest.mark.parametrize(
    "query, num_results, expected_output, http_code, error_msg",
    [
        (
            "invalid query",
            3,
            "Error: <HttpError 400 when requesting https://www.googleapis.com/customsearch/v1?q=invalid+query&cx "
            'returned "Invalid Value". Details: "Invalid Value">',
            400,
            "Invalid Value",
        ),
        (
            "invalid API key",
            3,
            "Error: The provided Google API key is invalid or missing.",
            403,
            "invalid API key",
        ),
    ],
)
def test_google_official_search_errors(
    query,
    num_results,
    expected_output,
    mock_googleapiclient,
    http_code,
    error_msg,
    config,
):
    class resp:
        def __init__(self, _status, _reason):
            self.status = _status
            self.reason = _reason

    response_content = {
        "error": {"code": http_code, "message": error_msg, "reason": "backendError"}
    }
    error = HttpError(
        resp=resp(http_code, error_msg),
        content=str.encode(json.dumps(response_content)),
        uri="https://www.googleapis.com/customsearch/v1?q=invalid+query&cx",
    )

    mock_googleapiclient.side_effect = error
    actual_output = google_official_search(query, config, num_results=num_results)
    assert actual_output == safe_google_results(expected_output)
