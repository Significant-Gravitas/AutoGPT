"""Unit tests for the shared _api.py layer: envelope decoding, result
row normalization, and the AnySearchClient request surface."""

from typing import Any, cast
from unittest import mock

import pytest
from pydantic import SecretStr

from backend.blocks.anysearch._api import (
    ANYSEARCH_API_URL,
    ANYSEARCH_CLIENT,
    AnySearchClient,
    result_from_dict,
    unwrap_envelope,
)
from backend.blocks.anysearch.test_helpers import TEST_CREDENTIALS
from backend.sdk import APIKeyCredentials
from backend.util.request import HTTPServerError


def test_unwrap_envelope_success():
    env = unwrap_envelope(
        {
            "code": 0,
            "message": "success",
            "request_id": "req-9",
            "data": {"results": []},
        }
    )
    assert env.data == {"results": []}
    assert env.request_id == "req-9"


def test_unwrap_envelope_api_error_carries_request_id():
    with pytest.raises(ValueError, match="unauthorized.*req-7"):
        unwrap_envelope(
            {
                "code": 401,
                "message": "unauthorized",
                "request_id": "req-7",
                "data": {},
            }
        )


def test_unwrap_envelope_sanitizes_error_code_names():
    """The marker match is deliberately broad - even the machine error
    name ``invalid_api_key`` trips it and is withheld (over-redaction is
    intended; request_id still carries the debugging hook)."""
    with pytest.raises(ValueError) as exc:
        unwrap_envelope(
            {"code": 401, "message": "invalid_api_key", "request_id": "r", "data": {}}
        )
    assert "invalid_api_key" not in str(exc.value)
    assert "details withheld" in str(exc.value)


def test_unwrap_envelope_non_dict_body():
    """A JSON body that isn't an object is a malformed response, not a
    crash outside the expected-error boundary."""
    with pytest.raises(ValueError, match="malformed AnySearch response"):
        unwrap_envelope(cast(dict[str, Any], ["not", "a", "dict"]))


def test_unwrap_envelope_missing_data():
    """code==0 with a non-dict data is a malformed success - the upstream
    "success" message must not masquerade as an API error."""
    with pytest.raises(ValueError, match="malformed AnySearch response"):
        unwrap_envelope({"code": 0, "message": None, "data": None})

    with pytest.raises(ValueError, match="malformed AnySearch response"):
        unwrap_envelope({"code": 0, "message": "success", "data": "oops"})


def test_unwrap_envelope_bool_code():
    # False == 0 in Python; the contract uses numeric codes only.
    with pytest.raises(ValueError):
        unwrap_envelope({"code": False, "data": {}})


@pytest.mark.parametrize(
    "field,value",
    [
        ("code", True),
        ("code", "0"),
        ("code", None),
        ("data", [1, 2]),
        ("message", 42),
        ("request_id", 99),
    ],
    ids=["code_bool", "code_str", "code_null", "data_list", "message_int", "rid_int"],
)
def test_unwrap_envelope_wrong_types(field, value):
    """Envelope fields declared by the contract keep their types - a
    wrong-typed value is malformed, not silently coerced."""
    response: dict[str, Any] = {
        "code": 0,
        "message": "success",
        "request_id": "req-t",
        "data": {"results": []},
    }
    response[field] = value
    with pytest.raises(ValueError, match="malformed"):
        unwrap_envelope(response)


def test_unwrap_envelope_missing_required_key():
    """A successful response missing data.results is a response error -
    not an empty result set - and the error carries the request_id."""
    with pytest.raises(ValueError, match="missing data.results.*req-5"):
        unwrap_envelope(
            {
                "code": 0,
                "message": "success",
                "request_id": "req-5",
                "data": {},
            },
            required="results",
        )


def test_unwrap_envelope_sanitizes_credential_leaks():
    """A 402-style message echoing credential material must not be
    forwarded - it is replaced by a generic message plus request_id."""
    with pytest.raises(ValueError) as exc:
        unwrap_envelope(
            {
                "code": 402,
                "message": "quota exhausted: username u1 password p2 api_key k3",
                "request_id": "req-402",
                "data": None,
            }
        )
    text = str(exc.value)
    assert "password" not in text
    assert "api_key" not in text
    for value in ("u1", "p2", "k3"):
        assert value not in text
    assert "details withheld" in text
    assert "req-402" in text


def test_unwrap_envelope_sanitizes_authorization_echo():
    """An upstream error echoing the Authorization/Bearer header is
    credential material - the whole message is withheld."""
    with pytest.raises(ValueError) as exc:
        unwrap_envelope(
            {
                "code": 401,
                "message": "rejected Authorization: Bearer leaked-key-value",
                "request_id": "req-auth",
                "data": {},
            }
        )
    text = str(exc.value)
    assert "leaked-key-value" not in text
    assert "Bearer" not in text
    assert "details withheld" in text
    assert "req-auth" in text


def test_result_from_dict_defaults():
    r = result_from_dict({"url": "https://x.test"})
    assert r.title == ""
    assert r.url == "https://x.test"
    assert r.snippet == ""
    assert r.content is None


def test_result_from_dict_rejects_non_dict():
    with pytest.raises(ValueError, match="malformed result row"):
        result_from_dict(cast(dict[str, Any], "not-a-dict"))


def _ok_response(payload: dict):
    resp = mock.Mock()
    resp.ok = True
    resp.status = 200
    resp.json.return_value = payload
    return resp


@pytest.mark.asyncio
async def test_client_search_posts_to_search_endpoint():
    client = AnySearchClient(TEST_CREDENTIALS)
    resp = _ok_response({"code": 0, "message": "success", "data": {"results": []}})
    client.requests.post = mock.AsyncMock(return_value=resp)

    out = await client.search({"query": "q", "max_results": 1})

    assert out["data"]["results"] == []
    call = client.requests.post.call_args
    assert call is not None
    args, kwargs = call
    assert args[0] == f"{ANYSEARCH_API_URL}/v1/search"
    assert kwargs["json"]["query"] == "q"
    assert client.requests.extra_headers == {
        "Authorization": "Bearer mock-anysearch-api-key",
        "X-Anysearch-Client": ANYSEARCH_CLIENT,
    }


@pytest.mark.asyncio
async def test_client_empty_api_key_sends_no_authorization():
    """An empty key means the anonymous tier: the request must carry no
    Authorization header at all (an empty Bearer token is never sent),
    while the attribution header still goes out."""
    creds = APIKeyCredentials(
        id="01234567-89ab-cdef-0123-456789abcdef",
        provider="anysearch",
        api_key=SecretStr(""),
        title="Empty key",
        expires_at=None,
    )
    client = AnySearchClient(creds)
    assert client.requests.extra_headers == {"X-Anysearch-Client": ANYSEARCH_CLIENT}

    resp = _ok_response({"code": 0, "message": "success", "data": {"results": []}})
    client.requests.post = mock.AsyncMock(return_value=resp)
    out = await client.search({"query": "q"})
    assert out["data"]["results"] == []


@pytest.mark.asyncio
async def test_client_missing_credentials_send_no_authorization():
    """The anonymous tier: no credential at all must still carry no
    Authorization header."""
    client = AnySearchClient(None)
    assert client.requests.extra_headers == {"X-Anysearch-Client": ANYSEARCH_CLIENT}

    resp = _ok_response({"code": 0, "message": "success", "data": {"results": []}})
    client.requests.post = mock.AsyncMock(return_value=resp)
    out = await client.search({"query": "q"})
    assert out["data"]["results"] == []


@pytest.mark.asyncio
async def test_client_non_ok_status_raises():
    """raise_for_status only rejects >=400; a stranded 3xx (no Location)
    must not reach response.json()."""
    client = AnySearchClient(TEST_CREDENTIALS)
    resp = mock.Mock()
    resp.ok = False
    resp.status = 301
    client.requests.post = mock.AsyncMock(return_value=resp)

    with pytest.raises(ValueError, match="HTTP 301"):
        await client.search({"query": "q"})

    with pytest.raises(ValueError, match="HTTP 301"):
        await client.extract("https://agpt.co")


@pytest.mark.asyncio
async def test_client_extract_posts_to_extract_endpoint():
    client = AnySearchClient(TEST_CREDENTIALS)
    resp = _ok_response({"code": 0, "message": "success", "data": {"title": "t"}})
    client.requests.post = mock.AsyncMock(return_value=resp)

    out = await client.extract("https://agpt.co")

    assert out["data"]["title"] == "t"
    call = client.requests.post.call_args
    assert call is not None
    args, kwargs = call
    assert args[0] == f"{ANYSEARCH_API_URL}/v1/extract"
    assert kwargs["json"] == {"url": "https://agpt.co"}


@pytest.mark.asyncio
async def test_client_domains_gets_catalog():
    client = AnySearchClient(None)
    resp = _ok_response({"code": 0, "message": "success", "data": {"domains": []}})
    client.requests.get = mock.AsyncMock(return_value=resp)

    out = await client.domains()

    assert out["data"]["domains"] == []
    call = client.requests.get.call_args
    assert call is not None
    args, _kwargs = call
    assert args[0] == f"{ANYSEARCH_API_URL}/v1/domains"


@pytest.mark.asyncio
async def test_client_sub_domains_repeats_domain_param():
    """``?domain=a&domain=b`` repeated query params - never comma-joined."""
    client = AnySearchClient(TEST_CREDENTIALS)
    resp = _ok_response({"code": 0, "message": "success", "data": {"domains": []}})
    client.requests.get = mock.AsyncMock(return_value=resp)

    await client.sub_domains(["code", "finance"])

    call = client.requests.get.call_args
    assert call is not None
    args, _kwargs = call
    url = args[0]
    assert url.startswith(f"{ANYSEARCH_API_URL}/v1/sub-domains?")
    assert "domain=code" in url
    assert "domain=finance" in url
    assert "," not in url


@pytest.mark.asyncio
async def test_client_sub_domains_requires_a_domain():
    client = AnySearchClient(TEST_CREDENTIALS)
    client.requests.get = mock.AsyncMock()
    with pytest.raises(ValueError, match="at least one domain"):
        await client.sub_domains([])
    client.requests.get.assert_not_called()


@pytest.mark.asyncio
async def test_client_catalog_non_ok_raises():
    """Catalog endpoints reject stranded redirects with their HTTP status."""
    client = AnySearchClient(None)
    resp = mock.Mock()
    resp.ok = False
    resp.status = 304
    client.requests.get = mock.AsyncMock(return_value=resp)

    with pytest.raises(ValueError, match="HTTP 304"):
        await client.domains()

    with pytest.raises(ValueError, match="HTTP 304"):
        await client.sub_domains(["code"])


@pytest.mark.asyncio
async def test_client_catalog_server_error_propagates():
    """An HTTPServerError raised by the transport still propagates unchanged."""
    client = AnySearchClient(None)
    client.requests.get = mock.AsyncMock(
        side_effect=HTTPServerError("HTTP 502 Error: Bad Gateway, Body: upstream", 502)
    )

    with pytest.raises(HTTPServerError):
        await client.domains()

    with pytest.raises(HTTPServerError):
        await client.sub_domains(["code"])
