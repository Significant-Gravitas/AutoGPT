import json

import pytest
from googleapiclient.errors import HttpError
from httplib2 import Response
from pydantic import SecretStr

from forge.utils.exceptions import ConfigurationError

from . import WebSearchComponent


@pytest.fixture
def web_search_component():
    component = WebSearchComponent()
    if component.config.google_api_key is None:
        component.config.google_api_key = SecretStr("test")
    if component.config.google_custom_search_engine_id is None:
        component.config.google_custom_search_engine_id = SecretStr("test")
    return component


@pytest.mark.parametrize(
    "query, expected_output",
    [("test", "test"), (["test1", "test2"], '["test1", "test2"]')],
)
@pytest.fixture
def test_safe_google_results(
    query, expected_output, web_search_component: WebSearchComponent
):
    result = web_search_component.safe_google_results(query)
    assert isinstance(result, str)
    assert result == expected_output


@pytest.fixture
def test_safe_google_results_invalid_input(web_search_component: WebSearchComponent):
    with pytest.raises(AttributeError):
        web_search_component.safe_google_results(123)  # type: ignore


@pytest.mark.parametrize(
    "query, num_results, expected_output_parts, return_value",
    [
        (
            "test",
            1,
            ("Result 1", "https://example.com/result1"),
            [{"title": "Result 1", "href": "https://example.com/result1"}],
        ),
        ("", 1, (), []),
        ("no results", 1, (), []),
    ],
)
def test_google_search(
    query,
    num_results,
    expected_output_parts,
    return_value,
    mocker,
    web_search_component: WebSearchComponent,
):
    mock_ddg = mocker.Mock()
    mock_ddg.return_value = return_value

    mocker.patch("forge.components.web.search.DDGS.text", mock_ddg)
    actual_output = web_search_component.web_search(query, num_results=num_results)
    for o in expected_output_parts:
        assert o in actual_output


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
    query,
    num_results,
    expected_output,
    search_results,
    mock_googleapiclient,
    web_search_component: WebSearchComponent,
):
    mock_googleapiclient.return_value = search_results
    actual_output = web_search_component.google(query, num_results=num_results)
    assert actual_output == web_search_component.safe_google_results(expected_output)


@pytest.mark.parametrize(
    "query, num_results, expected_error_type, http_code, error_msg",
    [
        (
            "invalid query",
            3,
            HttpError,
            400,
            "Invalid Value",
        ),
        (
            "invalid API key",
            3,
            ConfigurationError,
            403,
            "invalid API key",
        ),
    ],
)
def test_google_official_search_errors(
    query,
    num_results,
    expected_error_type,
    mock_googleapiclient,
    http_code,
    error_msg,
    web_search_component: WebSearchComponent,
):
    response_content = {
        "error": {"code": http_code, "message": error_msg, "reason": "backendError"}
    }
    error = HttpError(
        resp=Response({"status": http_code, "reason": error_msg}),
        content=str.encode(json.dumps(response_content)),
        uri="https://www.googleapis.com/customsearch/v1?q=invalid+query&cx",
    )

    mock_googleapiclient.side_effect = error
    with pytest.raises(expected_error_type):
        web_search_component.google(query, num_results=num_results)
