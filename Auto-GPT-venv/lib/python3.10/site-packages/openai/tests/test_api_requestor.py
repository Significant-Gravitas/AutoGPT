import json

import pytest
import requests
from pytest_mock import MockerFixture

from openai import Model
from openai.api_requestor import APIRequestor


@pytest.mark.requestor
def test_requestor_sets_request_id(mocker: MockerFixture) -> None:
    # Fake out 'requests' and confirm that the X-Request-Id header is set.

    got_headers = {}

    def fake_request(self, *args, **kwargs):
        nonlocal got_headers
        got_headers = kwargs["headers"]
        r = requests.Response()
        r.status_code = 200
        r.headers["content-type"] = "application/json"
        r._content = json.dumps({}).encode("utf-8")
        return r

    mocker.patch("requests.sessions.Session.request", fake_request)
    fake_request_id = "1234"
    Model.retrieve("xxx", request_id=fake_request_id)  # arbitrary API resource
    got_request_id = got_headers.get("X-Request-Id")
    assert got_request_id == fake_request_id


@pytest.mark.requestor
def test_requestor_open_ai_headers() -> None:
    api_requestor = APIRequestor(key="test_key", api_type="open_ai")
    headers = {"Test_Header": "Unit_Test_Header"}
    headers = api_requestor.request_headers(
        method="get", extra=headers, request_id="test_id"
    )
    assert "Test_Header" in headers
    assert headers["Test_Header"] == "Unit_Test_Header"
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test_key"


@pytest.mark.requestor
def test_requestor_azure_headers() -> None:
    api_requestor = APIRequestor(key="test_key", api_type="azure")
    headers = {"Test_Header": "Unit_Test_Header"}
    headers = api_requestor.request_headers(
        method="get", extra=headers, request_id="test_id"
    )
    assert "Test_Header" in headers
    assert headers["Test_Header"] == "Unit_Test_Header"
    assert "api-key" in headers
    assert headers["api-key"] == "test_key"


@pytest.mark.requestor
def test_requestor_azure_ad_headers() -> None:
    api_requestor = APIRequestor(key="test_key", api_type="azure_ad")
    headers = {"Test_Header": "Unit_Test_Header"}
    headers = api_requestor.request_headers(
        method="get", extra=headers, request_id="test_id"
    )
    assert "Test_Header" in headers
    assert headers["Test_Header"] == "Unit_Test_Header"
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test_key"
