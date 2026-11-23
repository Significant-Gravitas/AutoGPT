"""What redact_url() and redact_urls_in_text() mask, and what they keep.

Every credential route is covered and nothing else is touched, and a seeded
randomized test buries the credential among junk chosen to end a match early.
request_url_logging_test.py and credential_scrubbing_test.py check the same
masking end to end.
"""

import contextvars
import io
import logging
import random
import re
import time
from collections.abc import Iterator

import pytest
from starlette.routing import BaseRoute, Mount, Route, WebSocketRoute

from backend.api import rest_api
from backend.util import url_redaction
from backend.util.url_redaction import (
    REDACTED,
    CredentialHeaderFilter,
    RequestURLRedactionFilter,
    credentials_in_headers,
    credentials_in_url,
    is_credential_header,
    keep_request_credentials,
    mask_credential,
    mask_known,
    redact_query_string,
    redact_url,
    redact_urls_in_text,
)


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
        # Telegram takes a bot's token in the path of every call.
        (
            "https://api.telegram.org/bot123:AA-b_c/getMe",
            f"https://api.telegram.org/bot{REDACTED}/getMe",
        ),
        (
            "https://api.telegram.org/file/bot123:AA/photos/p.jpg",
            f"https://api.telegram.org/file/bot{REDACTED}/photos/p.jpg",
        ),
        # Its host in any case, and with its default port.
        (
            "https://API.TELEGRAM.ORG/bot123:AA-b_c/getMe",
            f"https://API.TELEGRAM.ORG/bot{REDACTED}/getMe",
        ),
        (
            "https://api.telegram.org:443/bot123:AA-b_c/getMe",
            f"https://api.telegram.org:443/bot{REDACTED}/getMe",
        ),
        # A user and password in the authority, but not an @ in a path or query.
        (
            "https://bot:pa55w0rd@proxy.example:8080/v1/x",
            f"https://{REDACTED}@proxy.example:8080/v1/x",
        ),
        # A password can hold an @, so the authority's last one ends it.
        (
            "https://user:pass@word@example.com/path",
            f"https://{REDACTED}@example.com/path",
        ),
        ("https://host/u/@me", "https://host/u/@me"),
        # A route's template names a credential without being one.
        ("/api/public/shared/{share_token}", "/api/public/shared/{share_token}"),
        (
            "/api/public/shared/chats/{share_token}/messages",
            "/api/public/shared/chats/{share_token}/messages",
        ),
        (
            "/api/integrations/{provider}/webhooks/{webhook_id}/ingress",
            "/api/integrations/{provider}/webhooks/{webhook_id}/ingress",
        ),
        ("/api/public/shared/{x}y", f"/api/public/shared/{REDACTED}"),
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
        (
            "could not connect to postgresql://app:hunter22@db:5432/app: refused",
            f"could not connect to postgresql://{REDACTED}@db:5432/app: refused",
        ),
        (
            "mail a@b.example from https://host/u/@me",
            "mail a@b.example from https://host/u/@me",
        ),
        # Telegram's bot token is in the path, with no query and no /api/.
        (
            "POST https://api.telegram.org/bot123:AA-b_c/getMe",
            f"POST https://api.telegram.org/bot{REDACTED}/getMe",
        ),
        (
            "POST https://Api.Telegram.Org:443/bot123:AA-b_c/getMe",
            f"POST https://Api.Telegram.Org:443/bot{REDACTED}/getMe",
        ),
        (
            "proxy https://user:pass@word@proxy.example:8080 refused",
            f"proxy https://{REDACTED}@proxy.example:8080 refused",
        ),
        # A path or host can hold brackets and parentheses.
        (
            "GET https://h/v1/models(test)?key=AIza-12345678",
            f"GET https://h/v1/models(test)?key={REDACTED}",
        ),
        ("dial http://[::1]?key=abcdefgh", f"dial http://[::1]?key={REDACTED}"),
        # Characters a client can send raw in a path.
        (
            "GET /api/foo|bar^baz\\qux?token=S3cret-12345678 HTTP/1.1",
            f"GET /api/foo|bar^baz\\qux?token={REDACTED} HTTP/1.1",
        ),
        (
            "GET /api/{share_token}?token=S3cret-12345678",
            f"GET /api/{{share_token}}?token={REDACTED}",
        ),
        ("Did it work? Yes.", "Did it work? Yes."),
        # Code that only looks like a query, as a traceback's source lines and
        # an exception's message quote it, is left as it is.
        (
            'PATTERN = re.compile(r"(?=[A-Z])\\w+")',
            'PATTERN = re.compile(r"(?=[A-Z])\\w+")',
        ),
        ("match (?P<key>[a-z]+)=(?P<value>.*)", "match (?P<key>[a-z]+)=(?P<value>.*)"),
        (
            "SELECT * FROM t WHERE id = ? AND kind=?",
            "SELECT * FROM t WHERE id = ? AND kind=?",
        ),
        ("what?ok=1", "what?ok=1"),
        # ...but a URL beside it is masked.
        (
            'bad pattern (?=x) for "https://h/v1/models?key=AIza-1"',
            f'bad pattern (?=x) for "https://h/v1/models?key={REDACTED}',
        ),
    ],
)
def test_redact_urls_in_text(text: str, expected: str) -> None:
    assert redact_urls_in_text(text) == expected


@pytest.mark.parametrize(
    "query, expected",
    [
        ("key=AIza-1&alt=json", f"key={REDACTED}&alt={REDACTED}"),
        ("flag&key=", f"flag&key={REDACTED}"),
        ("", ""),
    ],
)
def test_redact_query_string(query: str, expected: str) -> None:
    assert redact_query_string(query) == expected


_UUID = "0b5e55ed-8a1f-4e27-b9c3-d4527e19beef"


@pytest.mark.parametrize(
    "url, expected",
    [
        (f"http://h/api/public/shared/{_UUID}", {_UUID}),
        (f"/api/integrations/github/webhooks/{_UUID}/ingress?x=1", {_UUID}),
        (f"/ws?a=1&token={_UUID}", {_UUID}),
        # Any query value named like a credential, raw and decoded.
        (
            "/cb?code=4%2F0AfJohX-abcdefgh&state=s",
            {"4%2F0AfJohX-abcdefgh", "4/0AfJohX-abcdefgh"},
        ),
        # A name and value as the route reads them, form-decoded, and as sent.
        (
            "/cb?%74oken=abc+def+ghi+jkl+mno",
            {"abc+def+ghi+jkl+mno", "abc def ghi jkl mno"},
        ),
        # A crafted Host header doesn't hide the path behind it.
        (f"http://evil?/api/invitations/{_UUID}/accept", {_UUID}),
        # Too short to be a credential, and not a credential's place.
        ("/api/public/shared/s-1?token=t-1", set()),
        (f"/api/library/agents/{_UUID}?page={_UUID}", set()),
    ],
)
def test_credentials_in_url(url: str, expected: set[str]) -> None:
    assert credentials_in_url(url) == expected


def test_credentials_in_headers() -> None:
    session = "Zk3pQ9vT2mW8xR4sL7yN1bH6cJ0dF5gA.aB3cD4eF5gH6"
    headers = [
        ("cookie", f'_ga=GA1.1.123456789.1700000000; sid="{session}"; theme=dark'),
        ("authorization", "Bearer bt5Hn8Qr2Wx6Lk9Pz3Vm7Jc4"),
        ("x-telegram-bot-api-secret-token", "tg9Kd3Pw7Xq2Mz5Rb8Lf1Hs4"),
        ("x-api-key", "short"),
        ("user-agent", "Mozilla/5.0 (X11; Linux x86_64) Firefox/140.0"),
    ]
    assert credentials_in_headers(headers) == {
        session,
        "Bearer bt5Hn8Qr2Wx6Lk9Pz3Vm7Jc4",
        "bt5Hn8Qr2Wx6Lk9Pz3Vm7Jc4",
        "tg9Kd3Pw7Xq2Mz5Rb8Lf1Hs4",
    }


@pytest.mark.parametrize(
    "text, credentials, expected",
    [
        ("pre-abcdefXYZ-post", ("abcdef", "defXYZ"), f"pre-{REDACTED}-post"),
        # Found as str.replace finds it, so a credential overlapping itself
        # leaves its tail, which no issued credential does.
        ("AAAA", ("AAA",), f"{REDACTED}A"),
        ("xAAAAx", ("AAA", "x"), f"{REDACTED}{REDACTED}A{REDACTED}"),
        ("tok1tok2", ("tok1", "tok2"), REDACTED * 2),
        ("nothing here", ("absent",), "nothing here"),
    ],
)
def test_mask_known_masks_overlapping_credentials_as_one(
    text: str, credentials: tuple[str, ...], expected: str
) -> None:
    assert mask_known(text, credentials) == expected


def test_blank_query_values_take_no_credential_slots() -> None:
    """A query can repeat a name with blanks before the value a route reads."""
    token = "the-value-a-route-reads-123"
    url = "/ws?" + "token=&" * 40 + f"token={token}"
    assert token in credentials_in_url(url)


def test_a_cookie_is_read_as_the_route_reads_it() -> None:
    value = "quoted-session-value-123456"
    found = credentials_in_headers([("cookie", f'session_token= "{value}"')])
    assert value in found


def test_a_request_value_in_a_url_does_not_hide_its_credentials() -> None:
    """A value the request sent can match part of a URL a log line quotes, such
    as its host; the URL's own credentials are masked all the same."""
    record = logging.LogRecord(
        "x",
        logging.ERROR,
        __file__,
        1,
        "POST %s failed",
        ("https://api.telegram.org/bot123:AA-b_c/getMe",),
        None,
    )
    in_text = logging.LogRecord(
        "x",
        logging.ERROR,
        __file__,
        1,
        "POST https://api.telegram.org/bot123:AA-b_c/getMe failed",
        (),
        None,
    )

    def filtered() -> None:
        keep_request_credentials(("api.telegram.org",))
        for each in (record, in_text):
            RequestURLRedactionFilter().filter(each)

    contextvars.copy_context().run(filtered)

    for each in (record, in_text):
        assert "123:AA-b_c" not in each.getMessage()


def test_one_credential_repeated_past_the_cap_is_masked_where_it_occurs() -> None:
    """The cap counts the places where different credentials occur; one
    credential, however often it repeats, is replaced where it occurs."""
    credential = "ABCDEFGHIJKLMNOP"

    masked = mask_known(f"keep {credential} " * 2000, (credential,))

    assert masked == f"keep {REDACTED} " * 2000


def test_overlapping_credentials_past_the_cap_leave_no_tail() -> None:
    """Past the cap, a text is masked whole rather than credential by
    credential, which could leave the tail of one another covered."""
    first, second = "ABCDEFGHIJKLMNOP", "KLMNOPQRSTUVWXYZ"
    text = f"xx{first}QRSTUVWXYZ" * 1100

    assert "QRSTUVWXYZ" not in mask_known(text, (first, second))


def test_an_argument_logged_as_its_repr_is_masked_as_shown() -> None:
    """``%r`` shows what ``str`` may hide."""
    token = "repr-only-token-Zq81mN2p"

    class Hidden:
        def __str__(self) -> str:
            return "hidden"

        def __repr__(self) -> str:
            return f"Hidden(token={token!r})"

    record = logging.LogRecord(
        "x", logging.ERROR, __file__, 1, "Lookup %r failed", (Hidden(),), None
    )

    def filtered() -> None:
        keep_request_credentials((token,))
        RequestURLRedactionFilter().filter(record)

    contextvars.copy_context().run(filtered)

    assert token not in record.getMessage()


def test_an_argument_whose_repr_fails_is_masked_as_its_str() -> None:
    """``%s`` shows ``str``, which still has to be checked when ``repr``
    raises."""
    token = "str-only-token-Zq81mN2p"

    class Shown:
        def __str__(self) -> str:
            return f"Shown(token={token})"

        def __repr__(self) -> str:
            raise RuntimeError("no repr")

    record = logging.LogRecord(
        "x", logging.ERROR, __file__, 1, "Lookup %s failed", (Shown(),), None
    )

    def filtered() -> None:
        keep_request_credentials((token,))
        RequestURLRedactionFilter().filter(record)

    contextvars.copy_context().run(filtered)

    assert token not in record.getMessage()


_UNSHOWN = "unshown-secret-Zq81mN2p"


class _Unshowable:
    """A value neither ``str`` nor ``repr`` can show, whose failure quotes a
    secret that no request sent."""

    def __str__(self) -> str:
        raise RuntimeError(f"cannot show {_UNSHOWN}")

    def __repr__(self) -> str:
        raise RuntimeError(f"cannot show {_UNSHOWN}")


@pytest.mark.parametrize(
    "msg, args",
    [
        ("Lookup %s failed", (_Unshowable(),)),
        (f"GET https://api.example.test/v1?token={_UNSHOWN} %d", ("not a number",)),
    ],
    ids=["an argument that can't be shown", "a malformed record quoting a URL"],
)
def test_a_record_that_cant_be_formatted_is_not_logged(
    msg: str, args: tuple, capsys: pytest.CaptureFixture[str]
) -> None:
    """A handler reports a record it can't format with its arguments and the
    failure's text, either of which can hold what the filter couldn't check,
    so the record reaches it as a notice that keeps the format, masked."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RequestURLRedactionFilter())
    record = logging.LogRecord("x", logging.ERROR, __file__, 1, msg, args, None)

    handler.handle(record)
    handler.close()

    assert _UNSHOWN not in stream.getvalue() + capsys.readouterr().err
    assert "Not logged" in stream.getvalue()


def test_an_argument_shown_alike_by_str_and_repr_is_scanned_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scanned: list[str] = []
    scan = url_redaction._masked_text

    def counted(text: str, credentials: tuple[str, ...]) -> str:
        scanned.append(text)
        return scan(text, credentials)

    monkeypatch.setattr(url_redaction, "_masked_text", counted)

    assert not url_redaction._masks(12345, ("a-credential-long-enough",))
    assert scanned == ["12345"]


def test_a_typed_argument_keeps_its_format_when_masked() -> None:
    """A number that matches a credential is masked where the rendered line
    shows it, so ``%d`` still formats it."""
    number = 12345678901234567890
    record = logging.LogRecord(
        "x", logging.ERROR, __file__, 1, "Account %d failed", (number,), None
    )

    def filtered() -> None:
        keep_request_credentials((str(number),))
        RequestURLRedactionFilter().filter(record)

    contextvars.copy_context().run(filtered)

    assert record.getMessage() == f"Account {REDACTED} failed"


def test_mask_credential_keeps_only_a_prefix_of_a_long_one() -> None:
    assert mask_credential(_UUID) == "0b5e55ed..."
    assert mask_credential("short-secret") == REDACTED


def test_a_records_traceback_is_masked_as_its_message_is() -> None:
    try:
        raise RuntimeError(r"GET https://h/v1/models?key=AIza-1 failed (?=[A-Z])")
    except RuntimeError as error:
        record = logging.makeLogRecord(
            {
                "msg": "failed: %s",
                "args": (error,),
                "exc_info": (RuntimeError, error, error.__traceback__),
            }
        )
    assert RequestURLRedactionFilter().filter(record)
    rendered = logging.Formatter().format(record)
    assert "AIza-1" not in rendered
    assert f"models?key={REDACTED} failed" in rendered
    assert "in test_a_records_traceback_is_masked_as_its_message_is" in rendered
    # Code stays readable in the message, the traceback's source line and its
    # exception line.
    assert rendered.count("failed (?=[A-Z])") == 3


def test_credential_header_filter_passes_any_record() -> None:
    """A filter that raises fails the logging call, so any message passes."""
    for msg in ({"event": "x"}, ["x"], None, 42, "%s %s"):
        record = logging.makeLogRecord({"msg": msg, "args": ("a", "b")})
        assert CredentialHeaderFilter().filter(record)


@pytest.mark.parametrize(
    "name, masked",
    [
        ("Authorization", True),
        ("proxy-authorization", True),
        ("x-airtable-content-mac", True),
        ("Cookie", True),
        ("set-cookie", True),
        ("X-API-Key", True),
        ("x-goog-api-key", True),
        ("X-Telegram-Bot-Api-Secret-Token", True),
        ("X-Hub-Signature-256", True),
        ("Sec-WebSocket-Protocol", True),
        ("Host", False),
        ("User-Agent", False),
        ("Origin", False),
        ("Sec-WebSocket-Version", False),
    ],
)
def test_credential_header_filter(name: str, masked: bool) -> None:
    for direction in "<>":
        record = logging.makeLogRecord(
            {"msg": f"{direction} %s: %s", "args": (name, "v-1")}
        )
        assert CredentialHeaderFilter().filter(record)
        assert record.getMessage() == f"{direction} {name}: " + (
            REDACTED if masked else "v-1"
        )


# `_CREDENTIAL_NAME` also decides which headers `Requests` drops on a redirect
# to another origin (`backend/util/request.py`). A change to it for masking
# changes those too: run `request_redirect_credentials_test.py` with this file.
@pytest.mark.parametrize(
    "name",
    [
        "auth",
        "auth_token",
        "token2",
        "x-auth-key",
        "api_key",
        "apiKey",
        "apikey",
        "X-API-Key",
        "Authorization",
        "access_token",
        "accessToken",
        "client_secret",
        "clientSecret",
        "password",
        "__Secure-better-auth.session_token",
        "sessionid",
        "csrftoken",
        "X-CSRF-Token",
        "X-Hub-Signature-256",
        "X-Amz-Credential",
        "sig",
        "code",
        "state",
        "jwt",
        "set-cookie",
        "Sec-WebSocket-Protocol",
        "x-airtable-content-mac",
        # Glued names, as cookies and normalised headers often are.
        "PHPSESSID",
        "JSESSIONID",
        "SECRETKEY",
        "clientsecret",
        "privatekey",
        "accesskey",
        "passcode",
        "apitoken",
        "idtoken",
        "authkey",
        "oauth_verifier",
        "oauthState",
        "api_key2",
        # A bare anti-forgery cookie or parameter.
        "_csrf",
        # Header names as Sentry shows an ASGI scope's bytes.
        "b'cookie'",
        "b'authorization'",
        "b'x-api-key'",
    ],
)
def test_a_name_holding_a_credential_word_is_read_as_a_credential(name: str) -> None:
    """Names that must stay masked, whatever the matching becomes: glued ones
    and byte-form header names included."""
    value = "a-value-long-enough-to-count"

    assert is_credential_header(name)
    assert value in credentials_in_url(f"/x?{name}={value}")


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
        target, credential = _random_request_target(rng)
        masked = redact_url(target)
        assert _SECRET not in masked, target
        assert redact_url(masked) == masked, target
        # Found whole too, as the route reads it, so it can be masked wherever
        # the route passes it on.
        assert credential in credentials_in_url(target), target
        line = f'1.2.3.4:5 - "GET {target} HTTP/1.1" 200'
        masked_line = redact_urls_in_text(line)
        assert _SECRET not in masked_line, line
        assert redact_urls_in_text(masked_line) == masked_line, line


def test_masking_stays_linear_in_a_url_stacked_with_prefixes() -> None:
    """A client can send a URL made of thousands of prefixes, and the filter
    masks it on the event loop; quadratic work took seconds here."""
    url = "/api/invitations/x" * 50_000
    started = time.thread_time()
    assert redact_url(url) == f"/api/invitations/{REDACTED}" * 50_000
    # CPU time on this thread, so a runner that pauses the test doesn't count.
    assert time.thread_time() - started < 2


def test_masking_a_query_in_text_stays_linear_in_a_long_path() -> None:
    """A client sends the path an access log line quotes, and each "/" in it
    could start the path before a query; tried from each, the work grew with
    the square of its length, about half a second at 16 KB."""
    line = '1.2.3.4:5 - "GET /api' + "/a" * 32_768 + ' HTTP/1.1" 404'
    started = time.thread_time()
    assert redact_urls_in_text(line) == line
    # CPU time on this thread, so a runner that pauses the test doesn't count.
    assert time.thread_time() - started < 0.5


@pytest.mark.parametrize("length, size", [(256, 100_000), (16, 1_000_000)])
def test_masking_known_credentials_stays_linear_when_they_repeat_a_character(
    length: int, size: int
) -> None:
    """A request can send credentials made of one repeated character, each of
    which then matches at every place of a long run of it."""
    credentials = tuple("a" * (length - i) for i in range(16))
    started = time.thread_time()
    assert REDACTED in mask_known("a" * size, credentials)
    # CPU time on this thread, so a runner that pauses the test doesn't count.
    assert time.thread_time() - started < 0.5


def test_logging_a_long_path_stays_linear_whatever_credentials_it_repeats() -> None:
    """The access log quotes a client's path, and the same client chooses the
    credentials the line is masked for."""
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("1.2.3.4:5", "GET", "/" + "a" * 100_000, "1.1", 404),
        None,
    )

    def filtered() -> None:
        keep_request_credentials(tuple("a" * (256 - i) for i in range(16)))
        RequestURLRedactionFilter().filter(record)

    started = time.thread_time()
    contextvars.copy_context().run(filtered)
    assert time.thread_time() - started < 0.5
    assert REDACTED in record.getMessage()


def test_reading_a_urls_credentials_stays_linear() -> None:
    """A query of thousands of pairs named like a credential, each value running
    to an "&" far behind. Read whole for each, the work grows with the square of
    the query: about 10 s and gigabytes of copies on this one."""
    url = "/x" + "?key=" * 50_000 + "&a=b" * 50_000
    started = time.thread_time()
    assert credentials_in_url(url)
    # CPU time on this thread, so a runner that pauses the test doesn't count.
    assert time.thread_time() - started < 2


_PATH_PARAM = re.compile(r"{(\w+)(?::\w+)?}")
# The path parameters that are bearer credentials.
_CREDENTIAL_PARAMS = {"token", "share_token", "webhook_id"}


def _route_paths(routes: list[BaseRoute], prefix: str = "") -> Iterator[str]:
    for route in routes:
        if isinstance(route, Mount):
            yield from _route_paths(route.routes, prefix + route.path)
        elif isinstance(route, (Route, WebSocketRoute)):
            yield prefix + route.path


_SECRET = "SeCrEt-7f3a-c0ffee-42"
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
    "//api.telegram.org/bot{}/getMe",
    "//api.telegram.org/file/bot{}/photos/p.jpg",
    "//API.Telegram.org:443/bot{}/getMe",
]


def _random_request_target(rng: random.Random) -> tuple[str, str]:
    """A target, and its credential as the route reads it."""

    def junk() -> str:
        return "".join(rng.choices(_JUNK, k=rng.randint(0, 3)))

    pairs = [f"{junk()}={junk()}" for _ in range(rng.randint(0, 3))]
    if rng.random() < 0.5:
        credential = f"{junk()}{_SECRET}{junk()}"
        pairs.insert(rng.randint(0, len(pairs)), f"token={credential}")
        path = rng.choice(["/ws", "/api/email/unsubscribe", "/"])
    else:
        credential = _SECRET
        # A crafted path or Host header can stack another prefix in front.
        stacked = rng.choice(["", "", "/api/public/shared", "/api/invitations"])
        path = stacked + rng.choice(_CREDENTIAL_PATHS).format(_SECRET)
    # Sentry prefixes the scheme and the Host header, which the client controls.
    origin = f"http://h{junk()}" if rng.random() < 0.3 else ""
    query = "?" + "&".join(pairs) if pairs or rng.random() < 0.5 else ""
    return origin + path + query, credential
