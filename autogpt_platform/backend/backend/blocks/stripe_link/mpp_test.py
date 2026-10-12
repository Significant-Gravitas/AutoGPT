"""Tests for the MPP challenge parsing and credential construction.

The credential format is not documented; it was derived from `Credential
.serialize` in `mppx` and confirmed against a live MPP merchant, which
accepted it and settled a real payment. These tests pin the shape so a
refactor cannot silently break it.
"""

import base64
import json

import pytest

from backend.blocks.stripe_link._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.stripe_link.mpp import (
    StripeLinkGetPaymentChallengeBlock,
    StripeLinkMPPPayBlock,
    build_credential,
    decode_payment_request,
    parse_payment_challenges,
    select_stripe_challenge,
)

# Shape of a real climate.stripe.dev response: two challenges in one header.
TWO_CHALLENGE_HEADER = (
    'Payment id="abc", realm="merchant.example", method="tempo", intent="charge", '
    'request="eyJhIjoxfQ", description="Climate contribution", '
    'expires="2026-08-18T04:05:32.228Z", '
    'Payment id="def", realm="merchant.example", method="stripe", intent="charge", '
    'request="eyJiIjoyfQ", description="Climate contribution", '
    'expires="2026-08-18T04:05:32.231Z"'
)


def test_both_challenges_are_parsed_from_one_header():
    """A merchant can offer several methods; picking the wrong one fails to pay."""
    challenges = parse_payment_challenges(TWO_CHALLENGE_HEADER)

    assert [c["method"] for c in challenges] == ["tempo", "stripe"]
    assert challenges[1]["id"] == "def"
    assert challenges[1]["realm"] == "merchant.example"


def test_a_header_without_payment_challenges_yields_nothing():
    assert parse_payment_challenges('Bearer realm="example"') == []
    assert parse_payment_challenges("") == []


def test_payment_request_decodes_without_padding():
    """The `request` blob is base64url with padding stripped."""
    payload = {
        "amount": "100",
        "currency": "usd",
        "methodDetails": {"networkId": "p_1"},
    }
    encoded = (
        base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    )

    assert decode_payment_request(encoded) == payload


def test_credential_round_trips_to_the_expected_wire_shape():
    challenge = {
        "id": "def",
        "realm": "merchant.example",
        "method": "stripe",
        "intent": "charge",
        "request": "eyJiIjoyfQ",
        "description": "Climate contribution",
        "expires": "2026-08-18T04:05:32.231Z",
    }
    credential = build_credential(challenge, "spt_123")

    scheme, _, encoded = credential.partition(" ")
    assert scheme == "Payment"
    assert "=" not in encoded, "mppx emits unpadded base64url"

    decoded = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    assert decoded["payload"] == {"spt": "spt_123"}
    # `request` must survive byte-for-byte: the server HMAC-binds the challenge.
    assert decoded["challenge"]["request"] == "eyJiIjoyfQ"
    assert decoded["challenge"]["id"] == "def"


def test_unknown_challenge_fields_are_not_echoed_back():
    """Only the fields mppx keeps travel back, or verification fails."""
    challenge = {
        "id": "def",
        "realm": "m",
        "method": "stripe",
        "intent": "charge",
        "request": "eyJiIjoyfQ",
        "vendor_extension": "should-be-dropped",
    }
    encoded = build_credential(challenge, "spt_1").split(" ", 1)[1]
    decoded = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))

    assert "vendor_extension" not in decoded["challenge"]


async def run_pay(spend_request: dict) -> dict:
    block = StripeLinkMPPPayBlock()

    async def _fake_link(credentials, method, path, body=None):
        return spend_request

    block._link_api_request = _fake_link  # type: ignore[method-assign]
    inp = block.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "spend_request_id": "lsrq_test",
            "url": "https://merchant.example/api/buy",
        }
    )
    return {n: v async for n, v in block.run(inp, credentials=TEST_CREDENTIALS)}


@pytest.mark.asyncio
async def test_paying_with_an_unapproved_request_is_refused():
    """The token does not exist yet; hitting the merchant would just fail."""
    outputs = await run_pay({"status": "pending_approval"})

    assert "not approved" in outputs["error"]
    assert "status_code" not in outputs


@pytest.mark.asyncio
async def test_paying_with_a_card_request_explains_the_mismatch():
    """A card spend request has no SPT — say so instead of failing obscurely."""
    outputs = await run_pay({"status": "approved", "card": {"number": "4242"}})

    assert "shared_payment_token" in outputs["error"]


@pytest.mark.asyncio
async def test_challenge_block_reports_non_mpp_merchants():
    block = StripeLinkGetPaymentChallengeBlock()

    async def _probe(*_a, **_kw):
        return 200, ""

    block._probe = _probe  # type: ignore[method-assign]

    inp = block.Input.model_validate({"url": "https://shop.example/checkout"})
    outputs = {n: v async for n, v in block.run(inp)}

    assert outputs["supports_mpp"] is False
    assert "network_id" not in outputs


@pytest.mark.asyncio
async def test_challenge_block_ignores_a_402_without_a_stripe_method():
    """Onchain-only merchants answer 402 but we cannot pay them."""
    block = StripeLinkGetPaymentChallengeBlock()
    onchain_only = 'Payment id="abc", realm="m", method="tempo", intent="charge", request="eyJhIjoxfQ"'

    async def _probe(*_a, **_kw):
        return 402, onchain_only

    block._probe = _probe  # type: ignore[method-assign]

    inp = block.Input.model_validate({"url": "https://shop.example/checkout"})
    outputs = {n: v async for n, v in block.run(inp)}

    assert outputs["supports_mpp"] is False


class _FakeResponse:
    """Stands in for `backend.util.request.Response`."""

    def __init__(self, status: int, payload=None, headers=None, text=""):
        self.status = status
        self._payload = payload
        self.headers = headers or {}
        self._text = text

    @property
    def content(self) -> bytes:
        if self._payload is not None:
            return json.dumps(self._payload).encode()
        return self._text.encode()

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def text(self):
        return self._text


class _RecordingClient:
    """Returns queued responses and records every request made."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def request(
        self, method, url, *, json=None, headers=None, allow_redirects=True
    ):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "json": json,
                "headers": headers or {},
                "allow_redirects": allow_redirects,
            }
        )
        return self._responses.pop(0)


def _patch_requests(monkeypatch, client):
    from backend.blocks.stripe_link import mpp

    configurations = []

    def _factory(*args, **kwargs):
        configurations.append(kwargs)
        return client

    monkeypatch.setattr(mpp, "Requests", _factory)
    return configurations


CHALLENGE_HEADER = (
    'Payment id="ch_1", realm="merchant.example", method="stripe", '
    'intent="charge", request="eyJhIjoxfQ"'
)


@pytest.mark.asyncio
async def test_pay_attaches_the_credential_on_the_retry(monkeypatch):
    """The heart of MPP: probe unauthenticated, then retry carrying the token."""
    client = _RecordingClient(
        [
            _FakeResponse(402, headers={"www-authenticate": CHALLENGE_HEADER}),
            _FakeResponse(200, payload={"contribution_id": "pi_1"}),
        ]
    )
    configurations = _patch_requests(monkeypatch, client)

    block = StripeLinkMPPPayBlock()
    status, payload, payment_attempted = await block._pay_with_token(
        "spt_abc", "https://merchant.example/buy", "POST", {"amount": 100}, {}
    )

    assert status == 200
    assert payload == {"contribution_id": "pi_1"}
    assert payment_attempted is True
    assert configurations == [{"raise_for_status": False, "retry_max_attempts": 1}]
    assert len(client.calls) == 2
    assert all(call["allow_redirects"] is False for call in client.calls)
    # First hop must be unauthenticated — that is what elicits the challenge.
    assert "Authorization" not in client.calls[0]["headers"]
    credential = client.calls[1]["headers"]["Authorization"]
    assert credential.startswith("Payment ")
    assert (
        "spt_abc"
        in base64.urlsafe_b64decode(credential.split(" ", 1)[1] + "==").decode()
    )


@pytest.mark.asyncio
async def test_challenge_probe_does_not_retry_or_follow_redirects(monkeypatch):
    client = _RecordingClient(
        [_FakeResponse(402, headers={"www-authenticate": CHALLENGE_HEADER})]
    )
    configurations = _patch_requests(monkeypatch, client)

    block = StripeLinkGetPaymentChallengeBlock()
    status, header = await block._probe(
        "https://merchant.example/buy", "POST", {"amount": 100}
    )

    assert status == 402
    assert header == CHALLENGE_HEADER
    assert configurations == [{"raise_for_status": False, "retry_max_attempts": 1}]
    assert client.calls[0]["allow_redirects"] is False


@pytest.mark.asyncio
async def test_pay_does_not_retry_when_the_merchant_did_not_ask_for_payment(
    monkeypatch,
):
    """A 200 on the first hop means nothing is owed; sending the token anyway
    would leak a bearer credential for no reason."""
    client = _RecordingClient([_FakeResponse(200, payload={"ok": True})])
    _patch_requests(monkeypatch, client)

    block = StripeLinkMPPPayBlock()
    status, payload, payment_attempted = await block._pay_with_token(
        "spt_abc", "https://merchant.example/buy", "POST", {}, {}
    )

    assert status == 200 and payload == {"ok": True}
    assert payment_attempted is False
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_unauthenticated_success_is_not_reported_as_paid(monkeypatch):
    client = _RecordingClient([_FakeResponse(200, payload={"ok": True})])
    _patch_requests(monkeypatch, client)

    block = StripeLinkMPPPayBlock()

    async def _fake_link(credentials, method, path, body=None):
        return {
            "status": "approved",
            "shared_payment_token": {"id": "spt_abc"},
        }

    object.__setattr__(block, "_link_api_request", _fake_link)
    inp = block.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "spend_request_id": "lsrq_test",
            "url": "https://merchant.example/buy",
        }
    )

    outputs = {n: v async for n, v in block.run(inp, credentials=TEST_CREDENTIALS)}

    assert outputs == {"status_code": 200, "paid": False, "response": {"ok": True}}
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_pay_raises_when_402_offers_no_stripe_method(monkeypatch):
    onchain_only = (
        'Payment id="c", realm="m", method="tempo", intent="charge", request="e30"'
    )
    client = _RecordingClient(
        [_FakeResponse(402, headers={"www-authenticate": onchain_only})]
    )
    _patch_requests(monkeypatch, client)

    block = StripeLinkMPPPayBlock()
    with pytest.raises(RuntimeError, match="without a Stripe payment challenge"):
        await block._pay_with_token(
            "spt_abc", "https://merchant.example/buy", "POST", {}, {}
        )


@pytest.mark.asyncio
async def test_caller_headers_cannot_displace_the_credential(monkeypatch):
    """A caller-supplied Authorization must not survive into the paid retry."""
    client = _RecordingClient(
        [
            _FakeResponse(402, headers={"www-authenticate": CHALLENGE_HEADER}),
            _FakeResponse(200, payload={"ok": True}),
        ]
    )
    _patch_requests(monkeypatch, client)

    block = StripeLinkMPPPayBlock()
    await block._pay_with_token(
        "spt_abc",
        "https://merchant.example/buy",
        "POST",
        {},
        {"Authorization": "Bearer attacker", "Content-Type": "text/plain"},
    )

    retry_headers = client.calls[1]["headers"]
    assert retry_headers["Authorization"].startswith("Payment ")
    assert retry_headers["Content-Type"] == "application/json"


def test_oversized_request_blob_is_refused():
    """The blob is merchant-controlled; decoding it unbounded is a DoS vector."""
    with pytest.raises(ValueError, match="implausibly large"):
        decode_payment_request("A" * (17 * 1024))


def test_non_object_request_blob_is_refused():
    """`json.loads` can return a list; downstream `.get()` would explode."""
    encoded = base64.urlsafe_b64encode(b"[1,2,3]").decode().rstrip("=")
    with pytest.raises(ValueError, match="must decode to an object"):
        decode_payment_request(encoded)


def test_challenges_parse_regardless_of_field_order():
    """RFC 7235 does not fix auth-param order; Link merely happens to send
    `id` first. Splitting on `id=` would have merged these into one."""
    header = (
        'Payment realm="m", method="stripe", intent="charge", id="x", '
        'request="eyJhIjoxfQ", '
        'Payment method="tempo", id="y", realm="m", intent="charge", request="e30"'
    )
    challenges = parse_payment_challenges(header)

    assert [c["method"] for c in challenges] == ["stripe", "tempo"]
    assert challenges[0]["id"] == "x"
    assert select_stripe_challenge(header)["id"] == "x"  # type: ignore[index]


@pytest.mark.asyncio
async def test_probe_drops_a_caller_supplied_authorization(monkeypatch):
    """The first hop must be unauthenticated — an Authorization header the
    caller passed through would suppress the 402 and break the flow."""
    client = _RecordingClient(
        [
            _FakeResponse(402, headers={"www-authenticate": CHALLENGE_HEADER}),
            _FakeResponse(200, payload={"ok": True}),
        ]
    )
    _patch_requests(monkeypatch, client)

    block = StripeLinkMPPPayBlock()
    await block._pay_with_token(
        "spt_abc",
        "https://merchant.example/buy",
        "POST",
        {},
        {"Authorization": "Bearer caller-token"},
    )

    assert "Authorization" not in client.calls[0]["headers"]
    assert client.calls[1]["headers"]["Authorization"].startswith("Payment ")


@pytest.mark.asyncio
async def test_a_declined_payment_is_reported_as_unpaid(monkeypatch):
    """The money-sensitive case: a decline must never read as `paid`.

    The retry carries the token, so a 402/4xx on the second hop means the
    merchant refused the credential. Reporting that as success would tell an
    agent a purchase completed when nothing was bought.
    """
    client = _RecordingClient(
        [
            _FakeResponse(402, headers={"www-authenticate": CHALLENGE_HEADER}),
            _FakeResponse(402, payload={"error": "card_declined"}),
        ]
    )
    _patch_requests(monkeypatch, client)

    block = StripeLinkMPPPayBlock()
    status, payload, payment_attempted = await block._pay_with_token(
        "spt_abc", "https://merchant.example/buy", "POST", {"amount": 100}, {}
    )

    assert status == 402
    assert payload == {"error": "card_declined"}
    assert payment_attempted is True
    assert not (200 <= status < 300)


@pytest.mark.asyncio
async def test_a_lowercase_caller_authorization_cannot_duplicate_the_header(
    monkeypatch,
):
    """Stripping only the canonical spelling left a second Authorization.

    The probe dropped `authorization` case-insensitively but the retry merged
    over `Authorization`, so a lowercase caller header survived and the
    merchant received two — defeating the guarantee that callers cannot
    displace the credential.
    """
    client = _RecordingClient(
        [
            _FakeResponse(402, headers={"www-authenticate": CHALLENGE_HEADER}),
            _FakeResponse(200, payload={"ok": True}),
        ]
    )
    _patch_requests(monkeypatch, client)

    block = StripeLinkMPPPayBlock()
    await block._pay_with_token(
        "spt_abc",
        "https://merchant.example/buy",
        "POST",
        {"amount": 100},
        {"authorization": "Bearer caller-token"},
    )

    retry_headers = client.calls[1]["headers"]
    auth_headers = [k for k in retry_headers if k.lower() == "authorization"]
    assert len(auth_headers) == 1
    assert retry_headers[auth_headers[0]].startswith("Payment ")
    assert "caller-token" not in str(retry_headers)


def test_a_list_merchant_response_is_wrapped_into_a_dict():
    from backend.blocks.stripe_link.mpp import _json_or_text

    assert _json_or_text(_FakeResponse(200, payload=[1, 2, 3])) == {"data": [1, 2, 3]}


def test_a_non_json_merchant_response_falls_back_to_bounded_text():
    from backend.blocks.stripe_link.mpp import _json_or_text

    out = _json_or_text(_FakeResponse(500, text="y" * 5000))

    assert len(out["body"]) == 1000


@pytest.mark.asyncio
async def test_a_failed_probe_is_not_reported_as_a_non_mpp_merchant():
    """A 503 or a typo'd URL used to yield supports_mpp=false, which routes the
    agent into the virtual-card flow as though the merchant had been checked."""

    async def _probe(url, method, body):
        return 503, ""

    block = StripeLinkGetPaymentChallengeBlock()
    block._probe = _probe  # type: ignore[method-assign]

    inp = block.Input.model_validate({"url": "https://shop.example/checkout"})
    outputs = {n: v async for n, v in block.run(inp)}

    assert "supports_mpp" not in outputs
    assert "503" in outputs["error"]


@pytest.mark.asyncio
async def test_a_plain_success_still_reports_a_non_mpp_merchant():
    """A merchant that serves the request without demanding payment is a
    definitive 'not MPP', unlike a failed probe."""

    async def _probe(url, method, body):
        return 200, ""

    block = StripeLinkGetPaymentChallengeBlock()
    block._probe = _probe  # type: ignore[method-assign]

    inp = block.Input.model_validate({"url": "https://shop.example/checkout"})
    outputs = {n: v async for n, v in block.run(inp)}

    assert outputs["supports_mpp"] is False
    assert "error" not in outputs


@pytest.mark.asyncio
async def test_the_pay_path_does_not_send_an_include_for_the_token():
    """Regression: `?include=shared_payment_token` SUPPRESSES the field.

    Link returns `shared_payment_token` on the plain spend-request
    representation. Naming it as an include does not expand it — Link
    silently returns a trimmed object with the field absent, so the pay block
    reported "Spend request has no shared payment token ... likely created for
    a virtual card" on a request that was correctly created as a token
    request. `card` IS a valid include, which is why Retrieve Card worked and
    hid the asymmetry. Found only by paying a live merchant; the mocked tests
    agreed with the wrong assumption.
    """
    seen: dict = {}

    async def _fake(credentials, method, path, body=None):
        seen["path"] = path
        return {
            "status": "approved",
            "shared_payment_token": {"id": "spt_live_abc"},
        }

    block = StripeLinkMPPPayBlock()
    object.__setattr__(block, "_link_api_request", _fake)

    captured = {}

    async def _pay(spt, url, method, body, headers):
        captured["spt"] = spt
        return 200, {"ok": True}, True

    object.__setattr__(block, "_pay_with_token", _pay)

    inp = block.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "spend_request_id": "lsrq_test",
            "url": "https://merchant.example/buy",
        }
    )
    outputs = {n: v async for n, v in block.run(inp, credentials=TEST_CREDENTIALS)}

    assert "include" not in seen["path"]
    assert captured["spt"] == "spt_live_abc"
    assert outputs["paid"] is True


@pytest.mark.asyncio
async def test_a_malformed_challenge_is_not_reported_as_a_blocked_url():
    """Redaction has to name the actual cause.

    `Requests` signals an SSRF refusal with a bare `ValueError`, and so does
    every parse failure here — so a catch-all reported a merchant's malformed
    challenge blob as "Request blocked: the URL is not allowed", which is both
    wrong and unactionable.
    """

    async def _probe(url, method, body):
        return (
            402,
            'Payment id="c", realm="m", method="stripe", request="!!!not-base64!!!"',
        )

    block = StripeLinkGetPaymentChallengeBlock()
    block._probe = _probe  # type: ignore[method-assign]

    inp = block.Input.model_validate({"url": "https://shop.example/buy"})
    outputs = {n: v async for n, v in block.run(inp)}

    assert "not allowed" not in outputs.get("error", "")


@pytest.mark.asyncio
async def test_a_blocked_url_is_redacted(monkeypatch):
    """`Requests` names the host or resolved IP it refused. Useful in logs,
    but the `error` output is agent-visible and must not become a readout of
    what resolves on the internal network."""
    from backend.blocks.stripe_link import mpp

    class _Refusing:
        async def request(self, *a, **kw):
            raise ValueError(
                "Access to private IP 169.254.169.254 (metadata.internal) is blocked"
            )

    monkeypatch.setattr(mpp, "Requests", lambda *a, **kw: _Refusing())

    block = StripeLinkGetPaymentChallengeBlock()
    inp = block.Input.model_validate({"url": "https://metadata.internal/latest"})
    outputs = {n: v async for n, v in block.run(inp)}

    assert "169.254.169.254" not in outputs["error"]
    assert "metadata.internal" not in outputs["error"]
    assert "not allowed" in outputs["error"]


@pytest.mark.asyncio
async def test_a_402_without_any_challenge_header_is_payment_required():
    async def _probe(url, method, body):
        return 402, ""

    block = StripeLinkGetPaymentChallengeBlock()
    block._probe = _probe  # type: ignore[method-assign]

    inp = block.Input.model_validate({"url": "https://shop.example/buy"})
    outputs = {n: v async for n, v in block.run(inp)}

    assert outputs["supports_mpp"] is False
    # Wants payment, just not one we can make — the card flow is no fallback.
    assert outputs["payment_required"] is True


@pytest.mark.asyncio
async def test_an_onchain_only_merchant_is_distinguished_from_a_free_one():
    """Both yield supports_mpp=false; only one is a candidate for the card flow."""

    async def _onchain(url, method, body):
        return 402, 'Payment id="c", realm="m", method="tempo", request="eyJhIjoxfQ"'

    async def _free(url, method, body):
        return 200, ""

    block = StripeLinkGetPaymentChallengeBlock()
    inp = block.Input.model_validate({"url": "https://shop.example/buy"})

    block._probe = _onchain  # type: ignore[method-assign]
    onchain = {n: v async for n, v in block.run(inp)}
    block._probe = _free  # type: ignore[method-assign]
    free = {n: v async for n, v in block.run(inp)}

    assert onchain["supports_mpp"] is False and onchain["payment_required"] is True
    assert (
        free["supports_mpp"] is False and free.get("payment_required", False) is False
    )


def test_an_oversized_body_is_not_parsed():
    """The cap has to apply before deserialization, not after."""
    from backend.blocks.stripe_link.mpp import MAX_RESPONSE_BYTES, _json_or_text

    huge = _FakeResponse(200, payload={"blob": "x" * (MAX_RESPONSE_BYTES + 1024)})
    out = _json_or_text(huge)

    assert out["truncated"] is True
    assert len(out["body"]) <= 1000


# ---------------------------------------------------------------------------
# Body values reach the merchant with their JSON types intact
# ---------------------------------------------------------------------------
async def run_pay_against(
    responses: list[dict],
    *,
    body: dict | None = None,
) -> tuple[dict, list[dict], dict]:
    """Run the pay block against a scripted sequence of Link responses.

    Returns the block's outputs, what it sent to the merchant, and the call
    log, so a test can assert on the wire rather than on internals.
    """
    from backend.blocks.stripe_link.mpp import StripeLinkMPPPayBlock

    calls: list[dict] = []
    sent: dict = {}

    async def _fake_link(credentials, method, path, body=None):
        calls.append({"path": path})
        return responses[min(len(calls) - 1, len(responses) - 1)]

    async def _fake_pay(spt, url, method, body, headers):
        sent.update({"spt": spt, "body": body})
        return 200, {"ok": True}, True

    block = StripeLinkMPPPayBlock()
    object.__setattr__(block, "_link_api_request", _fake_link)
    object.__setattr__(block, "_pay_with_token", _fake_pay)

    payload = {
        "credentials": TEST_CREDENTIALS_INPUT,
        "spend_request_id": "lsrq_test",
        "url": "https://merchant.example/api/buy",
    }
    if body is not None:
        payload["body"] = body
    inp = block.Input.model_validate(payload)

    outputs = {n: v async for n, v in block.run(inp, credentials=TEST_CREDENTIALS)}
    return outputs, calls, sent


@pytest.mark.asyncio
async def test_number_shaped_body_values_are_sent_as_numbers():
    """The builder's key/value editor stringifies every value; merchants 400."""
    _, _, sent = await run_pay_against(
        [{"status": "approved", "shared_payment_token": {"id": "spt_x"}}],
        body={"amount": "100", "rate": "1.5", "sku": "0012345", "name": "board"},
    )

    assert sent["body"]["amount"] == 100
    assert isinstance(sent["body"]["amount"], int)
    assert sent["body"]["rate"] == 1.5
    assert sent["body"]["sku"] == "0012345", "leading zeros are not a number"
    assert sent["body"]["name"] == "board"
