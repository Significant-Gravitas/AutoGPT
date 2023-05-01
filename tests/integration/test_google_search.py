import pytest
from googleapiclient.errors import HttpError

from autogpt.commands.google_search import (
    google_official_search,
    google_search,
    safe_google_results,
)


@pytest.mark.parametrize(
    "input, expected_output",
    [("test", "test"), (["test1", "test2"], '["test1", "test2"]')],
)
def test_safe_google_results(input, expected_output):
    result = safe_google_results(input)
    assert isinstance(result, str)
    assert result == expected_output


def test_safe_google_results_invalid_input():
    with pytest.raises(AttributeError):
        safe_google_results(123)


@pytest.mark.parametrize(
    "query, num_results, expected_output",
    [
        (
            "test",
            1,
            '[\n    {\n        "title": "Result 1",\n        "link": "https://example.com/result1"\n    }\n]',
        ),
        ("", 1, "[]"),
        ("no results", 1, "[]"),
    ],
)
def test_google_search(query, num_results, expected_output, mocker):
    mock_ddg = mocker.Mock()
    if query == "test":
        mock_ddg.return_value = [
            {"title": "Result 1", "link": "https://example.com/result1"}
        ]
    else:
        mock_ddg.return_value = []

    mocker.patch("autogpt.commands.google_search.ddg", mock_ddg)
    actual_output = google_search(query, num_results=num_results)
    expected_output = safe_google_results(expected_output)
    assert actual_output == expected_output


@pytest.mark.parametrize(
    "query, num_results, expected_output",
    [
        (
            "test",
            3,
            [
                "http://example.com/result1",
                "http://example.com/result2",
                "http://example.com/result3",
            ],
        ),
        ("", 3, []),
        (
            "invalid query",
            3,
            "Error: <HttpError 400 when requesting https://www.googleapis.com/customsearch/v1?q=invalid+query&cx "
            'returned "Invalid Value". Details: "Invalid Value">',
        ),
        (
            "invalid API key",
            3,
            "Error: The provided Google API key is invalid or missing.",
        ),
    ],
)
def test_google_official_search(query, num_results, expected_output, mocker):
    mock_build = mocker.patch("googleapiclient.discovery.build")
    mock_service = mocker.Mock()
    mock_build.return_value = mock_service

    if query == "test":
        search_results = [
            {"link": "http://example.com/result1"},
            {"link": "http://example.com/result2"},
            {"link": "http://example.com/result3"},
        ]
        mock_service.cse().list().execute().get.return_value = search_results
    elif query == "":
        search_results = []
        mock_service.cse().list().execute().get.return_value = search_results
    elif query == "invalid query":

        class resp:
            status = "400"
            reason = "Invalid Value"

        error = HttpError(
            resp=resp(),
            content=b'{"error": {"code": 400, "message": "Invalid Value", "reason": "backendError"}}',
            uri="https://www.googleapis.com/customsearch/v1?q=invalid+query&cx",
        )
        mock_service.cse().list().execute().get.side_effect = error
    elif query == "invalid API key":

        class resp:
            status = "403"
            reason = "invalid API key"

        error = HttpError(
            resp=resp(),
            content=b'{"error": {"code": 403, "message": "invalid API key", "reason": "backendError"}}',
            uri="https://www.googleapis.com/customsearch/v1?q=invalid+api+ley&cx",
        )
        mock_service.cse().list().execute().get.side_effect = error

    actual_output = google_official_search(query, num_results=num_results)

    assert actual_output == safe_google_results(expected_output)
