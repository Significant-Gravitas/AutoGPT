"""What redact_url() and redact_urls_in_text() mask, and what they keep.

Every credential route is covered and nothing else is touched, and a seeded
randomized test buries the credential among junk chosen to end a match early.
request_url_logging_test.py checks the same masking end to end.
"""

import random
import re
import time
from collections.abc import Iterator

import pytest
from starlette.routing import BaseRoute, Mount, Route, WebSocketRoute

from backend.api import rest_api
from backend.util.url_redaction import REDACTED, redact_url, redact_urls_in_text


@pytest.mark.parametrize(
    "url, expected",
    [
        ("/ws?token=a.b.c", f"/ws?token={REDACTED}"),
        ("/x?token=a&page=2&&flag", f"/x?token={REDACTED}&page={REDACTED}&&flag"),
        ("/x?token=a=b;c=d", f"/x?token={REDACTED}"),
        ("/x?a=1?token=b", f"/x?a={REDACTED}"),
        ("/x?=bare", f"/x?={REDACTED}"),
        # uvicorn's WebSocket scope keeps a raw # in the query.
        ("/ws?a=1#&token=b", f"/ws?a={REDACTED}&token={REDACTED}"),
        # Sentry builds the URL from the Host header, which can hold a ?.
        (
            "http://evil?/api/public/shared/chats/s-1/messages",
            f"http://evil?/api/public/shared/chats/{REDACTED}/messages",
        ),
        # A prefix stacked in front of a real one doesn't hide the real one.
        (
            "http://x/api/invitations/api/public/shared/chats/s-1/messages",
            f"http://x/api/invitations/{REDACTED}/public/shared/chats/{REDACTED}"
            "/messages",
        ),
        # The query is masked first, so its result is checked for paths too.
        (
            "/ws?/api/integrations/a=x/y&/webhooks/t",
            f"/ws?/api/integrations/a={REDACTED}&/webhooks/{REDACTED}",
        ),
        # Characters that can end a URL in a message don't end this one.
        ("/ws?q=it's&token=a", f"/ws?q={REDACTED}&token={REDACTED}"),
        ('/ws?q="&token=a', f"/ws?q={REDACTED}&token={REDACTED}"),
        ("/ws?q=a b&token=c", f"/ws?q={REDACTED}&token={REDACTED}"),
        (
            "https://h:1/api/email/unsubscribe?token=dXNlcjpzaWc%3D",
            f"https://h:1/api/email/unsubscribe?token={REDACTED}",
        ),
        (
            "/api/integrations/github/webhooks/w-1/ingress",
            f"/api/integrations/github/webhooks/{REDACTED}/ingress",
        ),
        (
            "/api/integrations/webhooks/w-1/ping",
            f"/api/integrations/webhooks/{REDACTED}/ping",
        ),
        (
            "/api/platform-linking/tokens/t-1/confirm",
            f"/api/platform-linking/tokens/{REDACTED}/confirm",
        ),
        (
            "/api/platform-linking/user-tokens/t-1/confirm",
            f"/api/platform-linking/user-tokens/{REDACTED}/confirm",
        ),
        ("/api/public/shared/s-1", f"/api/public/shared/{REDACTED}"),
        (
            "/api/public/shared/s-1/files/f-1/download",
            f"/api/public/shared/{REDACTED}/files/f-1/download",
        ),
        (
            "/api/public/shared/chats/s-1/messages?limit=5",
            f"/api/public/shared/chats/{REDACTED}/messages?limit={REDACTED}",
        ),
        ("/api/invitations/i-1/accept", f"/api/invitations/{REDACTED}/accept"),
        # A route that doesn't exist still carries the token.
        ("/api/invitations/i-1", f"/api/invitations/{REDACTED}"),
        ("/api/invitations/pending-1/x", f"/api/invitations/{REDACTED}/x"),
        # A path prefix in front of the app.
        (
            "/backend/api/invitations/i-1/accept",
            f"/backend/api/invitations/{REDACTED}/accept",
        ),
        # Already masked stays the same, and IDs that grant nothing are kept.
        (f"/ws?token={REDACTED}", f"/ws?token={REDACTED}"),
        ("/api/orgs/o-1/invitations/i-1", "/api/orgs/o-1/invitations/i-1"),
        ("/api/invitations/pending", "/api/invitations/pending"),
        ("/api/integrations/github/webhooks", "/api/integrations/github/webhooks"),
        ("/api/library/agents/a-1", "/api/library/agents/a-1"),
    ],
)
def test_redact_url(url: str, expected: str) -> None:
    assert redact_url(url) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            '1.2.3.4:5 - "GET /api/invitations/i-1/decline?x=1 HTTP/1.1" 401',
            f'1.2.3.4:5 - "GET /api/invitations/{REDACTED}/decline?x={REDACTED}'
            ' HTTP/1.1" 401',
        ),
        # A raw quote is part of the query; only whitespace ends it.
        (
            '"GET /ws?q=it\'s&token=a HTTP/1.1"',
            f'"GET /ws?q={REDACTED}&token={REDACTED} HTTP/1.1"',
        ),
        (
            "acting as user u for requesting GET https://h/api/public/shared/s-1?x=1",
            "acting as user u for requesting GET "
            f"https://h/api/public/shared/{REDACTED}?x={REDACTED}",
        ),
        (
            '"GET /ws?a=1#&token=t HTTP/1.1"',
            f'"GET /ws?a={REDACTED}&token={REDACTED} HTTP/1.1"',
        ),
        ("Did it work? Yes.", "Did it work? Yes."),
    ],
)
def test_redact_urls_in_text(text: str, expected: str) -> None:
    assert redact_urls_in_text(text) == expected


def test_every_route_with_a_credential_in_its_path_is_masked_and_no_other() -> None:
    """Checks the masked prefixes against the REST app's real routes, so a new
    route with a token in its path fails here until it is covered."""
    for path in _route_paths(rest_api.app.routes):
        sample = _PATH_PARAM.sub(
            lambda match: (
                f"secret-{match[1]}" if match[1] in _CREDENTIAL_PARAMS else "value"
            ),
            path,
        )
        if _CREDENTIAL_PARAMS & set(_PATH_PARAM.findall(path)):
            assert "secret-" not in redact_url(sample), path
        else:
            assert redact_url(sample) == sample, path


@pytest.mark.parametrize("seed", range(5))
def test_no_request_target_keeps_its_credential_whatever_surrounds_it(
    seed: int,
) -> None:
    """Random request targets with the credential in a query value, in a path
    segment or both, among junk chosen to end a match early. Masked on their
    own and inside an access log line, the credential never survives, and
    masking again changes nothing."""
    rng = random.Random(seed)
    for _ in range(200):
        target = _random_request_target(rng)
        masked = redact_url(target)
        assert _SECRET not in masked, target
        assert redact_url(masked) == masked, target
        line = f'1.2.3.4:5 - "GET {target} HTTP/1.1" 200'
        masked_line = redact_urls_in_text(line)
        assert _SECRET not in masked_line, line
        assert redact_urls_in_text(masked_line) == masked_line, line


def test_masking_stays_linear_in_a_url_stacked_with_prefixes() -> None:
    """A client can send a URL made of thousands of prefixes, and the filter
    masks it on the event loop; quadratic work took seconds here."""
    url = "/api/invitations/x" * 50_000
    started = time.monotonic()
    assert redact_url(url) == f"/api/invitations/{REDACTED}" * 50_000
    assert time.monotonic() - started < 2


_PATH_PARAM = re.compile(r"{(\w+)(?::\w+)?}")
# The path parameters that are bearer credentials.
_CREDENTIAL_PARAMS = {"token", "share_token", "webhook_id"}


def _route_paths(routes: list[BaseRoute], prefix: str = "") -> Iterator[str]:
    for route in routes:
        if isinstance(route, Mount):
            yield from _route_paths(route.routes, prefix + route.path)
        elif isinstance(route, (Route, WebSocketRoute)):
            yield prefix + route.path


_SECRET = "SeCrEt-7f3a"
# A request target has no whitespace, and & always separates pairs; anything
# else may sit next to the credential, a raw # included.
_JUNK = ["", "a", "it's", '"', "?", "#", "=", ";", "/", "%26", "%3D", "[x]", "{"]
_CREDENTIAL_PATHS = [
    "/api/integrations/generic_webhook/webhooks/{}/ingress",
    "/api/integrations/webhooks/{}/ping",
    "/api/platform-linking/tokens/{}/confirm",
    "/api/platform-linking/user-tokens/{}/confirm",
    "/api/public/shared/{}",
    "/api/public/shared/chats/{}/messages",
    "/api/invitations/{}/accept",
    "/api/invitations/{}",
]


def _random_request_target(rng: random.Random) -> str:
    def junk() -> str:
        return "".join(rng.choices(_JUNK, k=rng.randint(0, 3)))

    pairs = [f"{junk()}={junk()}" for _ in range(rng.randint(0, 3))]
    if rng.random() < 0.5:
        pairs.insert(rng.randint(0, len(pairs)), f"token={junk()}{_SECRET}{junk()}")
        path = rng.choice(["/ws", "/api/email/unsubscribe", "/"])
    else:
        # A crafted path or Host header can stack another prefix in front.
        stacked = rng.choice(["", "", "/api/public/shared", "/api/invitations"])
        path = stacked + rng.choice(_CREDENTIAL_PATHS).format(_SECRET)
    # Sentry prefixes the scheme and the Host header, which the client controls.
    origin = f"http://h{junk()}" if rng.random() < 0.3 else ""
    query = "?" + "&".join(pairs) if pairs or rng.random() < 0.5 else ""
    return origin + path + query
