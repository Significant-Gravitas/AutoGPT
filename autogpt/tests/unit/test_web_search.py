import json
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from forge.components.web.search import SearchResult, WebSearchComponent
from forge.config import Config
from forge.utils.exceptions import ConfigurationError
from googleapiclient.errors import HttpError
from pytest_mock import MockerFixture

from autogpt.agents.agent import Agent

if TYPE_CHECKING:
    from googleapiclient._apis.customsearch.v1 import Search as GoogleSearch


@pytest.fixture
def web_search_component(agent: Agent):
    return agent.web_search


@pytest.mark.parametrize(
    "query, num_results, ddg_return_value, expected_output",
    [
        (
            "test",
            1,
            [{"title": "Result 1", "href": "https://example.com/result1"}],
            [SearchResult(title="Result 1", url="https://example.com/result1")],
        ),
        ("no results", 1, (), []),
    ],
)
def test_ddg_search(
    query: str,
    num_results: int,
    ddg_return_value: list[dict],
    expected_output: list[SearchResult],
    mocker: MockerFixture,
    web_search_component: WebSearchComponent,
):
    mock_ddg = mocker.Mock()
    mock_ddg.return_value = ddg_return_value

    mocker.patch("forge.components.web.search.DDGS.text", mock_ddg)
    actual_output = web_search_component.web_search(query, num_results=num_results)
    for o in expected_output:
        assert o in actual_output


@pytest.fixture
def mock_googleapiclient(mocker: MockerFixture):
    mock_build = mocker.patch("googleapiclient.discovery.build")
    mock_service = mocker.Mock()
    mock_build.return_value = mock_service
    return mock_service.cse().list().execute().get


@pytest.mark.parametrize(
    "query, num_results, google_return_value, expected_output",
    [
        (
            "test",
            3,
            [
                {"title": "Result 1", "link": "http://example.com/result1"},
                {"title": "Result 2", "link": "http://example.com/result2"},
                {"title": "Result 3", "link": "http://example.com/result3"},
            ],
            [
                SearchResult(title="Result 1", url="http://example.com/result1"),
                SearchResult(title="Result 2", url="http://example.com/result2"),
                SearchResult(title="Result 3", url="http://example.com/result3"),
            ],
        ),
    ],
)
def test_google_custom_search(
    query: str,
    num_results: int,
    google_return_value: "GoogleSearch",
    expected_output: list[SearchResult],
    config: Config,
    mock_googleapiclient: Mock,
    web_search_component: WebSearchComponent,
):
    config.google_api_key = "mock_api_key"
    config.google_custom_search_engine_id = "mock_search_engine_id"

    mock_googleapiclient.return_value = google_return_value
    actual_output = web_search_component.google(query, num_results=num_results)
    assert actual_output == expected_output


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
def test_google_custom_search_errors(
    query: str,
    num_results: int,
    expected_error_type: type[Exception],
    http_code: int,
    error_msg: str,
    config: Config,
    mock_googleapiclient: Mock,
    web_search_component: WebSearchComponent,
):
    config.google_api_key = "mock_api_key"
    config.google_custom_search_engine_id = "mock_search_engine_id"

    class resp:
        def __init__(self, _status, _reason):
            self.status = _status
            self.reason = _reason

    response_content = {
        "error": {"code": http_code, "message": error_msg, "reason": "backendError"}
    }
    error = HttpError(
        resp=resp(http_code, error_msg),  # type: ignore
        content=str.encode(json.dumps(response_content)),
        uri="https://www.googleapis.com/customsearch/v1?q=invalid+query&cx",
    )

    mock_googleapiclient.side_effect = error
    with pytest.raises(expected_error_type):
        web_search_component.google(query, num_results=num_results)
