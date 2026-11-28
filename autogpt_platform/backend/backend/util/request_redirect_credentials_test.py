"""The credentials a hop to another origin leaves behind: the headers that
carry one, by name and by the shape of the name, and the ``auth=`` and
``cookies=`` aiohttp turns into headers. A hop that stays on the origin keeps
them all.

Every test drives real local servers and asserts what each hop received."""

import pytest

from backend.util import request as request_module
from backend.util.request import Requests
from backend.util.request_test_helpers import (
    Received,
    bounded,
    done,
    local_requests,
    location,
    origins,
    redirect,
    without_body,
)
from backend.util.url_redaction import is_credential_header

# --- Credentials stay on their origin ---

# Dropped on a hop to another origin by name, whatever ``is_credential_header``
# comes to match: the three dropped before, the common API key and token
# headers, and names some providers' APIs give them.
NAMED_CREDENTIAL_HEADERS = [
    "Authorization",
    "Proxy-Authorization",
    "Cookie",
    "X-Api-Key",
    "Api-Key",
    "X-Goog-Api-Key",
    "X-Auth-Token",
    "X-Access-Token",
    "X-Amz-Security-Token",
    "Key",
    "Api_Key",
    "Profile-Key",
    "X-Access-Key",
]
# Dropped by the shape of their names alone, as a glued name or a name of a
# provider's own is.
CREDENTIAL_SHAPED_HEADERS = [
    "apikey",
    "Private-Token",
    "X-Store-Token",
    "X-Client-Secret",
    "X-Session-Id",
    "X-Subscription-Key",
    "X-Request-Signature",
    "X-Auth-Email",
    # A legacy MCP server's session: a credential for that server alone.
    "Mcp-Session-Id",
]
# Headers that carry no credential, which every hop keeps.
ORDINARY_HEADERS = {
    "X-Request-Id": "order-1",
    "Accept": "application/json",
    "X-Title": "Orders",
}


async def _through_a_307(
    client: Requests, origin: str, headers: dict[str, str]
) -> tuple[Received, Received]:
    """A POST that a 307 sends on, body and all, to *origin*'s next hop."""
    async with origins() as (first, second, received):
        first.routes["/start"] = redirect(307, location(second, origin))
        first.routes["/next"] = second.routes["/next"] = done
        response = await bounded(
            client.post(f"{first.base}/start", json={"order": 1}, headers=headers)
        )
    assert response.status == 200
    initial, following = received
    assert following.server == ("first" if origin == "same-origin" else "second")
    assert following.body == b'{"order": 1}'
    return initial, following


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "where", ["headers", "extra_headers"], ids=["call-headers", "client-headers"]
)
@pytest.mark.parametrize("header", NAMED_CREDENTIAL_HEADERS + CREDENTIAL_SHAPED_HEADERS)
async def test_a_hop_to_another_origin_drops_each_credential_header(
    header: str, where: str
):
    credential = {header: "credential-value"}
    client = local_requests(
        extra_headers=credential if where == "extra_headers" else None
    )
    call_headers = {**ORDINARY_HEADERS, **(credential if where == "headers" else {})}

    initial, following = await _through_a_307(client, "cross-origin", call_headers)

    # It did go out, so the next hop can show it was dropped.
    assert initial.headers[header.lower()] == "credential-value"
    assert header.lower() not in following.headers
    assert {k.lower(): v for k, v in ORDINARY_HEADERS.items()}.items() <= (
        following.headers.items()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("header", NAMED_CREDENTIAL_HEADERS)
async def test_the_named_credential_headers_are_dropped_without_the_name_rule(
    header: str, monkeypatch: pytest.MonkeyPatch
):
    """``is_credential_header`` is tuned for masking logs; a change there must
    not let a header known to carry a credential through."""
    monkeypatch.setattr(request_module, "is_credential_header", lambda name: False)

    initial, following = await _through_a_307(
        local_requests(), "cross-origin", {header: "credential-value"}
    )

    assert initial.headers[header.lower()] == "credential-value"
    assert header.lower() not in following.headers


@pytest.mark.asyncio
@pytest.mark.parametrize("header", NAMED_CREDENTIAL_HEADERS + CREDENTIAL_SHAPED_HEADERS)
async def test_a_hop_on_the_same_origin_keeps_every_header(header: str):
    headers = {**ORDINARY_HEADERS, header: "credential-value"}

    _, following = await _through_a_307(local_requests(), "same-origin", headers)

    assert {k.lower(): v for k, v in headers.items()}.items() <= (
        following.headers.items()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("named", [True, False], ids=["named", "not-named"])
async def test_a_caller_still_names_the_credential_headers_the_rule_does_not_know(
    named: bool,
):
    header = "X-Passphrase"
    assert not is_credential_header(header)
    client = local_requests(sensitive_headers=[header] if named else [])

    initial, following = await _through_a_307(
        client, "cross-origin", {header: "credential-value"}
    )

    assert initial.headers["x-passphrase"] == "credential-value"
    assert ("x-passphrase" in following.headers) is not named


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["same-origin", "cross-origin"])
@pytest.mark.parametrize("status", [301, 303, 307])
async def test_the_auth_and_cookies_kwargs_follow_a_hop_only_on_the_same_origin(
    status: int, origin: str
):
    """aiohttp builds ``Authorization`` and ``Cookie`` from these kwargs for
    every hop, so a hop that leaves the body behind keeps them as long as it
    stays on the origin, and loses them elsewhere."""
    async with origins() as (first, second, received):
        first.routes["/start"] = redirect(status, location(second, origin))
        first.routes["/next"] = second.routes["/next"] = done
        await bounded(
            local_requests().post(
                f"{first.base}/start",
                json={"order": 1},
                auth=("user", "pw"),
                cookies={"session": "abc"},
            )
        )

    initial, following = received
    assert initial.headers["authorization"].startswith("Basic ")
    assert initial.headers["cookie"] == "session=abc"
    carried = origin == "same-origin"
    assert ("authorization" in following.headers) is carried
    assert ("cookie" in following.headers) is carried


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "where", ["headers", "extra_headers"], ids=["call-headers", "client-headers"]
)
@pytest.mark.parametrize("status", [301, 303])
async def test_a_cross_origin_hop_that_leaves_the_body_behind_leaves_the_credentials_too(
    status: int, where: str
):
    """Where the two rules meet: a POST answered by a 301 or 303 that sends
    the next hop to another origin is a GET without the body, its
    Content-Type and the credentials."""
    credentials = {"X-Api-Key": "credential-value", "Authorization": "Bearer token"}
    sent = {**credentials, "Content-Type": "application/json"}
    client = local_requests(extra_headers=sent if where == "extra_headers" else None)
    call_headers = {**ORDINARY_HEADERS, **(sent if where == "headers" else {})}
    async with origins() as (first, second, received):
        first.routes["/start"] = redirect(status, location(second, "cross-origin"))
        first.routes["/next"] = second.routes["/next"] = done
        await bounded(
            client.post(f"{first.base}/start", json={"order": 1}, headers=call_headers)
        )

    initial, following = received
    assert {k.lower(): v for k, v in sent.items()}.items() <= initial.headers.items()
    assert (following.server, following.method) == ("second", "GET")
    assert without_body(following)
    assert not {"x-api-key", "authorization"} & following.headers.keys()
    assert {k.lower(): v for k, v in ORDINARY_HEADERS.items()}.items() <= (
        following.headers.items()
    )
