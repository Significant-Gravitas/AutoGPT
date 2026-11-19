"""Tests for the Sentry ``before_send`` filter in ``metrics.py``.

The filter exists to keep noise out of Sentry without dropping load-bearing
errors. We pin the "drop" list at the call-site level so a regression that
changes ``_before_send`` semantics — e.g. accidentally dropping all pika
ERRORs, or letting StreamLostError through — surfaces as a test failure
rather than a Sentry alert flood.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from collections.abc import Callable

import pytest
import sentry_sdk
from sentry_sdk.consts import DEFAULT_OPTIONS
from sentry_sdk.profiler.transaction_profiler import has_profiling_enabled
from sentry_sdk.serializer import serialize
from sentry_sdk.utils import event_from_exception, json_dumps

# Imported at module scope on purpose: AppProcess calls sentry_init() in its
# class body, so the guard has to hold at collection time, not just in a test.
import backend.util.process
from backend.api.utils.client_ip import CLIENT_IP_TOKEN_HEADER
from backend.util import metrics, tracing
from backend.util.exceptions import InsufficientBalanceError
from backend.util.metrics import (
    _FALKORDB_DRIVER_LOGGER,
    _FALKORDB_TEARDOWN_SIGNATURES,
    _PIKA_RECONNECT_LOGGERS,
    _PIKA_RECONNECT_SIGNATURES,
    _before_send,
)
from backend.util.settings import AppEnvironment, BehaveAs, Config
from backend.util.url_redaction import REDACTED


def _log_event(logger: str, message: str) -> dict:
    return {
        "logger": logger,
        "logentry": {"formatted": message, "message": message},
        "level": "error",
    }


class _RedisConnectionError(Exception):
    """Exception whose *type* module mimics redis — so tests can tell the old
    module-based scoping (checked ``exc_type.__module__``) apart from the new
    traceback-based scoping (checks the raising frame's module)."""

    __module__ = "redis.exceptions"


def _exc_info_raised_from(module: str, message: str, exc_type=ConnectionError):
    """Build an ``exc_info`` triple whose traceback contains a frame in
    ``module`` and whose exception is ``exc_type`` — lets tests control both
    the traceback origin (what the new scoping checks) and the exception
    type's module (what the old module-based scoping checked)."""
    namespace: dict = {"__name__": module, "_Exc": exc_type}
    exec("def _raise(msg):\n    raise _Exc(msg)", namespace)
    try:
        namespace["_raise"](message)
    except exc_type:
        return sys.exc_info()
    raise AssertionError("expected exception")


# ---------- pika reconnect noise → dropped ----------


def test_pika_streamlost_error_dropped() -> None:
    """AUTOGPT-SERVER-6JC: ``StreamLostError: Transport indicated EOF`` from
    ``pika.adapters.blocking_connection`` is benign reconnect noise."""
    for logger in _PIKA_RECONNECT_LOGGERS:
        evt = _log_event(logger, "StreamLostError: Transport indicated EOF")
        assert _before_send(evt, hint={}) is None, logger


def test_pika_socket_eof_dropped() -> None:
    """AUTOGPT-SERVER-6JD: ``Socket EOF`` from
    ``pika.adapters.utils.io_services_utils`` reconnect path."""
    evt = _log_event(
        "pika.adapters.utils.io_services_utils",
        "Socket EOF on fd=12",
    )
    assert _before_send(evt, hint={}) is None


def test_pika_connection_lost_dropped() -> None:
    """AUTOGPT-SERVER-6JE: ``connection_lost`` callback firing during a
    rolling deploy."""
    evt = _log_event(
        "pika.adapters.base_connection",
        "connection_lost: Stream connection lost: ConnectionResetError(...)",
    )
    assert _before_send(evt, hint={}) is None


def test_pika_transport_eof_dropped() -> None:
    """AUTOGPT-SERVER-6JF: ``Transport indicated EOF`` standalone string."""
    evt = _log_event("pika.adapters.blocking_connection", "Transport indicated EOF")
    assert _before_send(evt, hint={}) is None


# ---------- pika ERRORs that must still get through ----------


def test_pika_authentication_failure_kept() -> None:
    """A real auth failure on the AMQP connection is load-bearing and must
    NOT be filtered out by the reconnect-noise rule."""
    evt = _log_event(
        "pika.adapters.blocking_connection",
        "ProbableAuthenticationError: Server closed connection",
    )
    assert _before_send(evt, hint={}) is not None


def test_pika_channel_declare_error_kept() -> None:
    """PRECONDITION_FAILED on a queue declare (e.g. quorum-type mismatch) is
    a real bug, not reconnect noise — must be kept."""
    evt = _log_event(
        "pika.adapters.blocking_connection",
        "PRECONDITION_FAILED - inequivalent arg 'x-queue-type' for queue 'foo'",
    )
    assert _before_send(evt, hint={}) is not None


def test_non_pika_logger_with_streamlost_kept() -> None:
    """Reconnect signatures are only suppressed for the three known pika
    loggers; any other logger emitting the same string is kept (e.g. a
    custom wrapper that re-raises)."""
    evt = _log_event(
        "backend.data.rabbitmq",
        "StreamLostError: Transport indicated EOF",
    )
    assert _before_send(evt, hint={}) is not None


def test_pika_reconnect_signatures_cover_all_four_known_patterns() -> None:
    """Sanity check: the signatures list still covers all four AUTOGPT-
    SERVER-6JC/6JD/6JE/6JF patterns from the prod Sentry issues."""
    expected = {
        "streamlosterror",
        "transport indicated eof",
        "socket eof",
        "connection_lost",
    }
    assert expected == set(_PIKA_RECONNECT_SIGNATURES)


def test_before_send_scrubs_secrets_from_actual_exception_event() -> None:
    secrets = {
        "id": "FAKE-ID-SECRET-991",
        "access": "FAKE-ACCESS-SECRET-992",
        "refresh": "FAKE-REFRESH-SECRET-993",
        "bearer": "FAKE-BEARER-SECRET-994",
        "provider": "FAKE-PROVIDER-SECRET-995",
        "device_code": "FAKE-DEVICE-CODE-996",
    }
    payload = {
        "tokens": {
            "id_token": secrets["id"],
            "access_token": secrets["access"],
            "refresh_token": secrets["refresh"],
        },
        "safe": "safe-frame-value",
    }
    try:
        raise RuntimeError("materialization failed")
    except RuntimeError:
        event, hint = event_from_exception(
            sys.exc_info(),
            client_options=DEFAULT_OPTIONS,
        )

    event.update(
        {
            "extra": {"payload": payload},
            "breadcrumbs": {
                "values": [
                    {
                        "data": {
                            "message": f"Authorization: Bearer {secrets['bearer']}",
                            "safe": "safe-breadcrumb-value",
                        }
                    }
                ]
            },
            "contexts": {
                "codex": {
                    "provider_state": secrets["provider"],
                    "user_code": secrets["device_code"],
                    "safe": "safe-context-value",
                }
            },
            "request": {
                "data": {"access_token": secrets["access"]},
                "headers": {"Authorization": f"Bearer {secrets['bearer']}"},
            },
        }
    )

    scrubbed = _before_send(event, hint)

    assert scrubbed is event
    serialized = json.dumps(scrubbed, default=str)
    for name, secret in secrets.items():
        assert secret not in serialized, name
    assert "safe-frame-value" in serialized
    assert "safe-breadcrumb-value" in serialized
    assert "safe-context-value" in serialized


def test_before_send_output_survives_the_sdk_transport() -> None:
    """The SDK runs ``before_send`` on the already-serialized event and then
    ``json_dumps`` it for the envelope. A scrubbed event that is not plain JSON
    is dropped there as an internal SDK error, so it never reaches Sentry.

    A frame local named ``session`` (as in ``download_with_fresh_session``) and
    an ``authorization`` request header are enough to trigger that."""

    def download_with_fresh_session():
        session = "aiohttp-client-session"
        if session:
            raise RuntimeError("Response payload is not completed")

    try:
        download_with_fresh_session()
    except RuntimeError:
        event, hint = event_from_exception(
            sys.exc_info(), client_options=DEFAULT_OPTIONS
        )
    event["request"] = {
        "method": "GET",
        "url": "http://backend/api/workspace/files/f/download",
        "headers": {"authorization": "Bearer FAKE-TOKEN-1", "accept": "*/*"},
    }

    scrubbed = _before_send(serialize(event), hint)

    assert scrubbed is not None
    body = json.loads(json_dumps(scrubbed))
    raising_frame = body["exception"]["values"][0]["stacktrace"]["frames"][-1]
    assert raising_frame["vars"]["session"] == "[Filtered]"
    assert body["request"]["headers"]["authorization"] == "[Filtered]"
    assert body["request"]["headers"]["accept"] == "*/*"
    assert body["exception"]["values"][0]["value"] == (
        "Response payload is not completed"
    )


_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJlLWJ5dGVz"


@pytest.mark.parametrize(
    "message, filtered",
    [
        (f"token {_JWT}", True),
        (f"base64-{_JWT}", True),
        ("eyJhbGciOiJIUzI1NiJ9 and nothing after", False),
    ],
)
def test_before_send_filters_a_jwt_wherever_its_run_starts(
    message: str, filtered: bool
) -> None:
    scrubbed = _before_send(_log_event("backend.api.rest_api", message), {})

    assert scrubbed is not None
    assert (scrubbed["logentry"]["formatted"] == "[Filtered]") is filtered


@pytest.mark.parametrize("name", metrics._SECRET_NAMES)
def test_before_send_filters_a_secret_under_its_name_however_it_is_spelled(
    name: str,
) -> None:
    for spelled in (name, name.upper(), name.title(), name.replace("_", "-")):
        for assignment in (f"{spelled}=hunter22", f'"{spelled}": "hunter22"'):
            message = f"Request failed with {assignment}"
            scrubbed = _before_send(_log_event("backend.api.rest_api", message), {})
            assert scrubbed is not None
            assert scrubbed["logentry"]["formatted"] == "[Filtered]", assignment


@pytest.mark.parametrize("unset", ["None", "null", "''", '""'])
def test_before_send_keeps_a_repr_whose_secret_fields_are_unset(unset: str) -> None:
    """A model's repr shows every field, the unset ones too."""
    message = f"Preset(id='p-1', webhook_id={unset}, share_token={unset})"
    scrubbed = _before_send(_log_event("backend.api.rest_api", message), {})
    assert scrubbed is not None
    assert scrubbed["logentry"]["formatted"] == message
    held = f"Preset(id='p-1', webhook_id={unset}, secret='hunter22')"
    scrubbed = _before_send(_log_event("backend.api.rest_api", held), {})
    assert scrubbed is not None
    assert scrubbed["logentry"]["formatted"] == "[Filtered]"


@pytest.mark.parametrize(
    "message",
    [
        '{"password": null, "name": "x"}',
        "Login(password=None, name='x')",
        "Preset(webhook_id=None, created=datetime.datetime(2026, 1, 1, 0, 0))",
        "User(password=None, status=<Status.ACTIVE: 'active'>, logins=3)",
    ],
)
def test_before_send_keeps_a_secret_left_unset_before_the_next_field(
    message: str,
) -> None:
    scrubbed = _before_send(_log_event("backend.api.rest_api", message), {})
    assert scrubbed is not None
    assert scrubbed["logentry"]["formatted"] == message


@pytest.mark.parametrize(
    "message",
    [
        "password=None.",
        "secret=None-example",
        "password=null-abc123",
        # In free text a comma can be part of the value.
        "password=None,hunter22",
        "login failed: password=None, hunter22 rejected",
        # Shaped like a repr's next field, but its value isn't one.
        "login failed: password=None, name=hunter22 rejected",
    ],
)
def test_before_send_filters_a_secret_that_only_starts_like_an_unset_one(
    message: str,
) -> None:
    scrubbed = _before_send(_log_event("backend.api.rest_api", message), {})
    assert scrubbed is not None
    assert scrubbed["logentry"]["formatted"] == "[Filtered]"


def test_before_send_reads_the_sdks_annotations_within_the_budget() -> None:
    """``_meta`` repeats each key a cut value sat under; its keys are read only
    while the budget lasts, and past it the annotations are dropped whole."""
    key = "x" * 40
    meta = {"extra": {"runs": {key: {"": {"len": 9000}}}}}

    def event(filler: int) -> dict:
        return {"extra": {"filler": "y" * filler}, "_meta": copy.deepcopy(meta)}

    kept = _before_send(event(1000), {})
    assert kept is not None and kept["_meta"] == meta
    left_out = _before_send(event(metrics._SENTRY_SCRUB_BUDGET - 20), {})
    assert left_out is not None and "_meta" not in left_out


def test_before_send_charges_a_query_as_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A query is charged its full length before it is read, so one past the
    budget is filtered unread."""
    read: list[str] = []
    redact = metrics.redact_query_string
    monkeypatch.setattr(
        metrics,
        "redact_query_string",
        lambda query: read.append(query) or redact(query),
    )
    event = {
        "extra": {"filler": "y" * (metrics._SENTRY_SCRUB_BUDGET - 100)},
        "spans": [{"data": {"http.query": "token=" + "z" * 5000}}],
    }

    scrubbed = _before_send(event, {})

    assert scrubbed is not None
    assert scrubbed["spans"][0]["data"]["http.query"] == "[Filtered]"
    assert read == []


def test_a_request_value_in_a_url_does_not_hide_its_credentials_from_sentry() -> None:
    """A value the request sent can match part of a URL an event quotes, such
    as its host; the URL's own credentials are masked all the same."""
    event = {
        "request": {"url": "http://backend/lookup?token=api.telegram.org"},
        "message": "POST https://api.telegram.org/bot123:AA-b_c/getMe failed",
    }

    scrubbed = _before_send(event, {})

    assert scrubbed is not None
    assert "123:AA-b_c" not in json.dumps(scrubbed)


def test_before_send_stays_linear_whatever_credentials_the_request_repeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request can send credentials made of one repeated character, and put
    long runs of it in what its error reports."""
    credentials = tuple("a" * (256 - i) for i in range(16))
    monkeypatch.setattr(metrics, "request_credentials", lambda: credentials)
    event = {"extra": {f"run{i}": "a" * 8192 for i in range(60)}}

    started = time.thread_time()
    scrubbed = _before_send(event, {})
    # CPU time on this thread, so a runner that pauses the test doesn't count;
    # about 0.14 s here with branch coverage on, and over 6 s for work growing
    # with the square of the input.
    assert time.thread_time() - started < 3
    assert scrubbed is not None and scrubbed["extra"]["run0"] != "a" * 8192


@pytest.mark.parametrize("send", [_before_send, metrics._before_send_transaction])
def test_a_credential_in_an_unmatched_routes_path_is_masked_everywhere(
    monkeypatch: pytest.MonkeyPatch, send: Callable[[dict, dict], dict | None]
) -> None:
    """A request no route matched names its transaction after its raw path."""
    bearer = "bt5Hn8Qr2Wx6Lk9Pz3Vm7Jc4eR1"
    monkeypatch.setattr(metrics, "request_credentials", lambda: (bearer,))
    event = {
        "type": "transaction",
        "transaction": f"/zzz/{bearer}",
        "transaction_info": {"source": "url"},
        "request": {"url": f"http://h/zzz/{bearer}"},
    }

    scrubbed = send(event, {})

    assert scrubbed is not None
    assert bearer not in json.dumps(scrubbed)
    assert scrubbed["transaction"] == f"/zzz/{REDACTED}"


@pytest.mark.parametrize(
    "transaction_info, kept",
    [
        ({"source": "route"}, True),
        ({"source": "custom"}, True),
        ({"source": "view"}, True),
        ({"source": "component"}, True),
        ({"source": "task"}, True),
        ({"source": "url"}, False),
        ({}, False),
        (None, False),
    ],
)
def test_a_transaction_is_masked_only_when_its_name_is_a_raw_path(
    monkeypatch: pytest.MonkeyPatch, transaction_info: dict | None, kept: bool
) -> None:
    """A name code set keeps its name, whatever a request sends; a raw path,
    which a request no route matched gives, is masked by the request's
    values, and so is a name whose source the SDK didn't say."""
    name = "/api/v1/graphs/a-credential-long-enough/execute"
    monkeypatch.setattr(metrics, "request_credentials", lambda: (name,))
    event: dict = {"type": "transaction", "transaction": name}
    if transaction_info is not None:
        event["transaction_info"] = transaction_info

    scrubbed = metrics._before_send_transaction(event, {})

    assert (scrubbed["transaction"] == name) is kept


def test_a_matched_routes_transaction_keeps_its_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request can send a route's own template as a credential; the route's
    events keep their name, and the copy elsewhere is masked."""
    template = "/api/v1/graphs/{graph_id}/execute"
    monkeypatch.setattr(metrics, "request_credentials", lambda: (template,))
    event = {
        "type": "transaction",
        "transaction": template,
        "transaction_info": {"source": "route"},
        "extra": {"sent": template},
    }

    scrubbed = metrics._before_send_transaction(event, {})

    assert scrubbed["transaction"] == template
    assert scrubbed["extra"]["sent"] == REDACTED


def test_an_outgoing_requests_fragment_is_masked_whole() -> None:
    """A fragment can be a bare token, which no query rule reads."""
    token = "frag3ment-token-Zq81mN2p"
    event = {
        "spans": [
            {"data": {"http.fragment": token, "http.query": "page=2"}},
            {"data": {"http.fragment": ""}},
        ]
    }

    scrubbed = _before_send(event, {})

    assert scrubbed is not None
    assert scrubbed["spans"][0]["data"]["http.fragment"] == REDACTED
    assert scrubbed["spans"][0]["data"]["http.query"] == f"page={REDACTED}"
    assert scrubbed["spans"][1]["data"]["http.fragment"] == ""


def test_an_unmatched_routes_transaction_is_masked_as_a_url() -> None:
    """Read as a whole URL, so its query is masked whatever comes before it."""
    secret = "S3cret-query-value-123"
    event = {
        "type": "transaction",
        "transaction": f'/a"?token={secret}',
        "transaction_info": {"source": "url"},
    }

    scrubbed = metrics._before_send_transaction(event, {})

    assert secret not in json.dumps(scrubbed)


def test_before_send_reads_an_event_whose_request_is_not_a_mapping() -> None:
    """An event that breaks the SDK's shape is still sent, not dropped."""
    event = {"request": "GET /lookup", "transaction_info": "url", "message": "x"}
    assert _before_send(event, {}) == event


def test_before_send_scans_a_run_of_jwt_starts_in_linear_time() -> None:
    message = "-eyJ" * 20_000
    started = time.thread_time()
    scrubbed = _before_send(_log_event("backend.api.rest_api", message), {})
    # A pattern entered at every eyJ takes about 10 s on this input.
    assert time.thread_time() - started < 1.0
    assert scrubbed is not None
    assert scrubbed["logentry"]["formatted"] == message


def test_before_send_searches_no_more_than_its_budget(monkeypatch) -> None:
    searched: list[int] = []
    pattern = metrics._EMBEDDED_SECRET_ASSIGNMENT

    class _Recording:
        def search(self, text: str):
            searched.append(len(text))
            return pattern.search(text)

    monkeypatch.setattr(metrics, "_EMBEDDED_SECRET_ASSIGNMENT", _Recording())
    chunk = "x" * metrics._SENTRY_MAX_VALUE_LENGTH
    crumbs = metrics._SENTRY_SCRUB_BUDGET // len(chunk) + 5
    event = _log_event("backend.api.rest_api", "api_key=FAKE-SECRET-1")
    event["breadcrumbs"] = {"values": [{"message": chunk} for _ in range(crumbs)]}
    frame = {"function": "handle_request", "vars": {"request": chunk}}
    event["exception"] = {"values": [{"stacktrace": {"frames": [frame]}}]}

    scrubbed = _before_send(event, {})

    assert scrubbed is not None
    assert sum(searched) <= metrics._SENTRY_SCRUB_BUDGET
    assert scrubbed["logentry"]["formatted"] == "[Filtered]"
    # A frame sits deeper than any breadcrumb, and is still searched first.
    assert frame == {"function": "handle_request", "vars": {"request": chunk}}
    messages = [crumb["message"] for crumb in scrubbed["breadcrumbs"]["values"]]
    assert set(messages) == {chunk, "[Filtered]"}


def _put_in_breadcrumbs(event: dict, chunks: list[str]) -> None:
    event["breadcrumbs"] = {"values": [{"message": chunk} for chunk in chunks]}


def _read_breadcrumbs(event: dict) -> list[str]:
    return [crumb["message"] for crumb in event["breadcrumbs"]["values"]]


def _put_in_frames(event: dict, chunks: list[str]) -> None:
    frames = [{"vars": {"local": chunk}} for chunk in chunks]
    event["exception"] = {"values": [{"stacktrace": {"frames": frames}}]}


def _read_frames(event: dict) -> list[str]:
    frames = event["exception"]["values"][0]["stacktrace"]["frames"]
    return [frame["vars"]["local"] for frame in frames]


@pytest.mark.parametrize(
    "put, read",
    [(_put_in_breadcrumbs, _read_breadcrumbs), (_put_in_frames, _read_frames)],
    ids=["breadcrumbs", "frames"],
)
def test_before_send_spends_its_budget_on_the_newest_entries(put, read) -> None:
    count = metrics._SENTRY_SCRUB_BUDGET // metrics._SENTRY_MAX_VALUE_LENGTH + 4
    chunks = [
        f"{index:04d}".ljust(metrics._SENTRY_MAX_VALUE_LENGTH, "x")
        for index in range(count)
    ]
    event = _log_event("backend.api.rest_api", "nothing secret")
    put(event, chunks)

    scrubbed = _before_send(event, {})

    assert scrubbed is not None
    values = read(scrubbed)
    # Searched and kept from the newest end, filtered from the oldest, in place.
    assert values[0] == "[Filtered]"
    first_kept = next(i for i, value in enumerate(values) if value != "[Filtered]")
    assert values[first_kept:] == chunks[first_kept:]
    assert set(values[:first_kept]) == {"[Filtered]"}


def test_before_send_transaction_keeps_every_span_id_past_its_budget() -> None:
    """A trace whose span IDs or timestamps were filtered would fall apart."""
    ids = [f"{index:016x}" for index in range(3)]
    started = "2026-11-18T09:00:00.123456Z"
    spans = [
        {
            "span_id": span_id,
            "parent_span_id": "b" * 16,
            "start_timestamp": started,
            "description": "GET /x",
        }
        for span_id in ids
    ]
    # A name that leaves less of the budget than one span ID would take.
    name = "x" * (metrics._SENTRY_SCRUB_BUDGET - len("transaction") - 8)
    event_id = "c" * 32
    event = {
        "type": "transaction",
        "transaction": name,
        # The event's own, after the name: read once the budget is spent.
        "event_id": event_id,
        "start_timestamp": started,
        "timestamp": started,
        "spans": spans,
    }

    scrubbed = metrics._before_send_transaction(event, {})

    assert scrubbed["transaction"] == name
    assert scrubbed["event_id"] == event_id
    assert scrubbed["start_timestamp"] == scrubbed["timestamp"] == started
    assert [span["span_id"] for span in scrubbed["spans"]] == ids
    assert {span["parent_span_id"] for span in scrubbed["spans"]} == {"b" * 16}
    assert {span["start_timestamp"] for span in scrubbed["spans"]} == {started}
    assert scrubbed["spans"][-1]["description"] == "[Filtered]"


def test_before_send_scrubs_a_payloads_field_named_like_a_trace_id() -> None:
    """Only the SDK's own records keep their IDs unsearched, even a value in
    their format: here a request's credential that looks like a trace ID."""
    token = "0123456789abcdef0123456789abcdef"
    payload = {
        "trace_id": token,
        "span_id": {"api_key": "k"},
        "event_id": "Bearer abcdefghijklmnop",
    }
    event = {
        "request": {"url": f"http://backend/api/public/shared/{token}"},
        "extra": {"payload": payload},
    }

    scrubbed = _before_send(event, {})

    assert scrubbed is not None
    assert scrubbed["extra"]["payload"] == {
        "trace_id": "[REDACTED]",
        "span_id": {"api_key": "[Filtered]"},
        "event_id": "[Filtered]",
    }


def test_before_send_masks_a_quoted_urls_query_but_not_code_like_one() -> None:
    message = r"pattern (?=[A-Z]) failed for https://h/v1/models?key=AIza-12345678"
    event = {
        "exception": {
            "values": [
                {
                    "value": message,
                    "stacktrace": {
                        "frames": [
                            {"context_line": '    re.compile(r"(?P<k>\\w+)=(?=x)")'}
                        ]
                    },
                }
            ]
        }
    }

    scrubbed = _before_send(event, {})

    assert scrubbed is not None
    exception = scrubbed["exception"]["values"][0]
    assert exception["value"] == (
        r"pattern (?=[A-Z]) failed for https://h/v1/models?key=[REDACTED]"
    )
    frame = exception["stacktrace"]["frames"][0]
    assert frame["context_line"] == '    re.compile(r"(?P<k>\\w+)=(?=x)")'


def test_before_send_keeps_untyped_balance_message() -> None:
    try:
        raise RuntimeError("Third-party API reported insufficient balance")
    except RuntimeError:
        exc_info = sys.exc_info()

    assert _before_send({"level": "error"}, hint={"exc_info": exc_info}) is not None


def test_before_send_drops_typed_insufficient_balance_error() -> None:
    try:
        raise InsufficientBalanceError(
            message="New producer wording without legacy keywords",
            user_id="user-1",
            balance=0,
            amount=1,
        )
    except InsufficientBalanceError:
        exc_info = sys.exc_info()

    assert _before_send({"level": "error"}, hint={"exc_info": exc_info}) is None


def test_before_send_keeps_wrapper_around_insufficient_balance_error() -> None:
    try:
        try:
            raise InsufficientBalanceError(
                message="New producer wording without legacy keywords",
                user_id="user-1",
                balance=0,
                amount=1,
            )
        except InsufficientBalanceError as error:
            raise RuntimeError("Unexpected execution wrapper") from error
    except RuntimeError:
        exc_info = sys.exc_info()

    assert _before_send({"level": "error"}, hint={"exc_info": exc_info}) is not None


# ---------- FalkorDB connection-teardown noise → dropped ----------


def test_falkordb_buffer_is_closed_log_dropped() -> None:
    """SENTRY-1387: ``Buffer is closed`` logged by graphiti-core's FalkorDB
    driver is a benign connection-teardown race — a query racing the cache
    eviction close or a per-request ``driver.close()``."""
    evt = _log_event(
        _FALKORDB_DRIVER_LOGGER,
        "Error executing FalkorDB query: Buffer is closed.\nMATCH (n) RETURN n\n{}",
    )
    assert _before_send(evt, hint={}) is None


def test_falkordb_connection_closed_by_server_log_dropped() -> None:
    """The sibling teardown message from the same race — the driver docstring
    pairs it with ``Buffer is closed``."""
    evt = _log_event(
        _FALKORDB_DRIVER_LOGGER,
        "Error executing FalkorDB query: Connection closed by server.",
    )
    assert _before_send(evt, hint={}) is None


def test_falkordb_buffer_is_closed_exc_from_graphiti_dropped() -> None:
    """If the re-raised teardown error reaches Sentry as an exception (a caller
    that doesn't swallow it), a ``Buffer is closed`` raised from the graphiti
    FalkorDB driver is still benign and must be dropped."""
    exc_info = _exc_info_raised_from(_FALKORDB_DRIVER_LOGGER, "Buffer is closed.")
    assert _before_send({"level": "error"}, hint={"exc_info": exc_info}) is None


def test_main_redis_connection_closed_exc_kept() -> None:
    """A genuine ``Connection closed by server`` from the platform's main redis
    (NOT raised from the graphiti driver) is a real incident and must NOT be
    swallowed by the teardown-noise rule. Uses a redis-module exception type so
    the OLD module-based scoping would have dropped it — this test fails if the
    traceback-based narrowing is reverted."""
    exc_info = _exc_info_raised_from(
        "redis.asyncio.connection",
        "Connection closed by server.",
        _RedisConnectionError,
    )
    assert _before_send({"level": "error"}, hint={"exc_info": exc_info}) is not None


def test_falkordb_real_query_error_kept() -> None:
    """A genuine Cypher/query failure from the same driver logger is load-
    bearing and must NOT be filtered out by the teardown-noise rule."""
    evt = _log_event(
        _FALKORDB_DRIVER_LOGGER,
        "Error executing FalkorDB query: Invalid input 'RETRUN': expected...",
    )
    assert _before_send(evt, hint={}) is not None


def test_falkordb_buffer_is_closed_from_other_logger_kept() -> None:
    """The teardown signatures are only suppressed for the graphiti FalkorDB
    driver logger; the same string from any other logger is kept."""
    evt = _log_event(
        "backend.data.redis_client",
        "Buffer is closed.",
    )
    assert _before_send(evt, hint={}) is not None


def test_falkordb_teardown_signatures_cover_known_patterns() -> None:
    """Sanity check: the teardown-signature list still covers both messages the
    graphiti FalkorDB driver docstring pairs together."""
    expected = {"buffer is closed", "connection closed by server"}
    assert expected == set(_FALKORDB_TEARDOWN_SIGNATURES)


# ---------- pytest runs must not reach Sentry ----------

_FAKE_DSN = "https://key@o1.ingest.us.sentry.io/1"


def _spy_on_sentry_init(monkeypatch) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(metrics, "_sentry_init", lambda **kw: calls.append(kw))
    monkeypatch.setattr(metrics.settings.secrets, "sentry_dsn", _FAKE_DSN)
    return calls


def test_no_sentry_client_is_active_under_pytest() -> None:
    """End-to-end. AppProcess runs sentry_init() in its class body, so this
    module's import above is what a live client here would have come from."""
    assert backend.util.process.AppProcess
    assert sentry_sdk.get_client().is_active() is False


def test_sentry_init_skipped_at_collection_time(monkeypatch) -> None:
    """pytest only sets PYTEST_CURRENT_TEST once a test item runs, so the
    import-time call that AppProcess makes is covered by sys.modules alone."""
    calls = _spy_on_sentry_init(monkeypatch)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    metrics.sentry_init()

    assert calls == []


def test_sentry_init_runs_outside_pytest(monkeypatch) -> None:
    calls = _spy_on_sentry_init(monkeypatch)
    monkeypatch.delitem(sys.modules, "pytest")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    metrics.sentry_init()

    assert len(calls) == 1
    assert calls[0]["dsn"] == _FAKE_DSN


def test_sentry_init_samples_through_the_shared_sampler(monkeypatch) -> None:
    """Probes, scrapes and WebSocket connections are dropped by the sampler;
    see tracing_test for its rules."""
    calls = _spy_on_sentry_init(monkeypatch)
    monkeypatch.delitem(sys.modules, "pytest")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    metrics.sentry_init()

    assert calls[0]["traces_sampler"] is tracing.traces_sampler


def test_sentry_init_sends_no_trace_headers_to_third_parties(monkeypatch) -> None:
    """Only the RPC client attaches trace headers (service_test covers it), so a
    block calling a user-supplied URL never hands it Sentry's trace or key."""
    calls = _spy_on_sentry_init(monkeypatch)
    monkeypatch.delitem(sys.modules, "pytest")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    metrics.sentry_init()

    assert calls[0]["trace_propagation_targets"] == []


def test_sentry_init_profiles_continuously_not_per_transaction(monkeypatch) -> None:
    calls = _spy_on_sentry_init(monkeypatch)
    monkeypatch.delitem(sys.modules, "pytest")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    metrics.sentry_init()

    options = {**DEFAULT_OPTIONS, **calls[0]}
    assert not has_profiling_enabled(options)
    assert options["profile_session_sample_rate"] == 1.0
    assert options["profile_lifecycle"] == "trace"


def test_sentry_init_cuts_long_values_before_they_are_scrubbed(monkeypatch) -> None:
    calls = _spy_on_sentry_init(monkeypatch)
    monkeypatch.delitem(sys.modules, "pytest")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    metrics.sentry_init()

    assert calls[0]["max_value_length"] == metrics._SENTRY_MAX_VALUE_LENGTH


def test_a_large_frame_local_reaches_the_event_truncated(monkeypatch) -> None:
    """A failed virus scan's traceback holds the upload in several frames, and the
    SDK's full repr of each took seconds and a gigabyte for a 90 MB file."""
    calls = _spy_on_sentry_init(monkeypatch)
    monkeypatch.delitem(sys.modules, "pytest")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    metrics.sentry_init()

    def write_file(content: bytes) -> None:
        raise RuntimeError("Virus scanning failed")

    try:
        write_file(b"x" * 5_000_000)
    except RuntimeError:
        event, _ = event_from_exception(
            sys.exc_info(), client_options={**DEFAULT_OPTIONS, **calls[0]}
        )

    raising_frame = event["exception"]["values"][0]["stacktrace"]["frames"][-1]
    assert len(raising_frame["vars"]["content"]) < 2_000


def test_a_long_local_is_masked_in_one_window_however_its_credentials_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A credential repeated through a long value shortens it wherever it is
    masked, and the longest kept credential sets how far a cut one can reach.
    One window of the value is read all the same: the head is shorter instead."""
    longest = "L0ng-" * 80
    repeated = "r3peated-Credential-0123456789"
    monkeypatch.setattr(metrics, "request_credentials", lambda: (longest, repeated))
    read: list[int] = []
    mask_known = metrics.mask_known

    def counted(text: str, credentials: tuple[str, ...]) -> str:
        read.append(len(text))
        return mask_known(text, credentials)

    monkeypatch.setattr(metrics, "mask_known", counted)

    for value in (repeated * 100_000, (repeated * 100_000).encode()):
        preview = metrics._bounded_repr(value)
        assert preview is not None and f"{len(value):,} long" in preview
        assert repeated[:8] not in preview and REDACTED in preview
    assert read == [metrics._MAX_LOCAL_REPR + 2 * len(longest)] * 2


@pytest.mark.parametrize("inside", range(8, 31, 5))
def test_a_local_cut_inside_a_credential_keeps_none_of_it(
    monkeypatch: pytest.MonkeyPatch, inside: int
) -> None:
    """The credentials masked before it shorten the window, which can bring
    one that the window cuts, `inside` characters in, into the head."""
    credential = "cUt-Credential-0123456789abcdef"
    monkeypatch.setattr(metrics, "request_credentials", lambda: (credential,))
    window = metrics._MAX_LOCAL_REPR + 2 * len(credential)
    before = credential * 20
    filler = "x" * (window - len(before) - inside)
    value = before + filler + credential + "y" * 5_000

    preview = metrics._bounded_repr(value)

    assert preview is not None and f"{REDACTED}x" in preview
    assert not any(credential[i : i + 8] in preview for i in range(len(credential) - 7))


def test_a_long_bytearray_local_keeps_its_type() -> None:
    preview = metrics._bounded_repr(bytearray(b"z" * 5_000))
    assert preview is not None and preview.startswith("bytearray(b'zzz")


def test_both_event_bounds_hold_through_one_trip_to_before_send(monkeypatch) -> None:
    """The SDK applies ``custom_repr`` and ``max_value_length`` in the one serialize
    pass before ``before_send``, so the scrubber only ever searches bounded text."""
    calls = _spy_on_sentry_init(monkeypatch)
    monkeypatch.delitem(sys.modules, "pytest")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    metrics.sentry_init()
    options = {**DEFAULT_OPTIONS, **calls[0]}

    def write_file(content: bytes) -> None:
        raise RuntimeError("Virus scanning failed")

    try:
        write_file(b"x" * 5_000_000)
    except RuntimeError:
        event, _ = event_from_exception(sys.exc_info(), client_options=options)

    event.update(_log_event("backend.api.rest_api", "api_key=FAKE-SECRET-1"))
    event["extra"] = {"transcript": "y" * 1_000_000}
    scrubbed = _before_send(
        serialize(
            event,
            max_value_length=options["max_value_length"],
            custom_repr=options["custom_repr"],
        ),
        {},
    )

    assert scrubbed is not None
    # custom_repr cut the frame local before repr() built a 5 MB copy, so it reports
    # the original length; without it the value arrives as a max_value_length slice.
    local = scrubbed["exception"]["values"][0]["stacktrace"]["frames"][-1]["vars"][
        "content"
    ]
    assert local.endswith("... (5,000,000 long)")
    assert len(local) < metrics._SENTRY_MAX_VALUE_LENGTH
    # max_value_length cut the long innocent value, so the scrubber searched it and
    # kept it; unbounded it would exceed the budget and be dropped as [Filtered].
    assert scrubbed["extra"]["transcript"] != "[Filtered]"
    assert len(scrubbed["extra"]["transcript"]) <= metrics._SENTRY_MAX_VALUE_LENGTH
    # And the budget the two bounds left over still caught the secret.
    assert scrubbed["logentry"]["formatted"] == "[Filtered]"


def test_sentry_init_skipped_in_subprocess_spawned_by_pytest(monkeypatch) -> None:
    """A spawned service subprocess does not inherit sys.modules, but does
    inherit PYTEST_CURRENT_TEST from the environment."""
    calls = _spy_on_sentry_init(monkeypatch)
    monkeypatch.delitem(sys.modules, "pytest")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "backend/util/metrics_test.py::t (call)")

    metrics.sentry_init()

    assert calls == []


# ---------- release and environment tagging ----------


def _sentry_init_options(monkeypatch) -> dict:
    """The options sentry_init() hands the SDK once past the pytest guard."""
    calls = _spy_on_sentry_init(monkeypatch)
    monkeypatch.delitem(sys.modules, "pytest")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    metrics.sentry_init()

    assert len(calls) == 1
    return calls[0]


def test_sentry_init_reports_the_deployed_commit_as_release(monkeypatch) -> None:
    monkeypatch.setattr(metrics.settings.config, "sentry_release", "0123abcd")

    assert _sentry_init_options(monkeypatch)["release"] == "0123abcd"


def test_sentry_init_trims_the_release(monkeypatch) -> None:
    monkeypatch.setattr(metrics.settings.config, "sentry_release", "  0123abcd\n")

    assert _sentry_init_options(monkeypatch)["release"] == "0123abcd"


@pytest.mark.parametrize("sha", ["", "   ", "\n"])
def test_sentry_init_reports_no_release_for_a_blank_sha(monkeypatch, sha) -> None:
    """A build made without the SENTRY_RELEASE build arg bakes in an empty
    value, which has to read as no release rather than a blank one."""
    monkeypatch.setattr(metrics.settings.config, "sentry_release", sha)

    assert _sentry_init_options(monkeypatch)["release"] is None


def test_sentry_init_environment_override_wins(monkeypatch) -> None:
    monkeypatch.setattr(metrics.settings.config, "sentry_environment", "pr-123")
    monkeypatch.setattr(metrics.settings.config, "app_env", AppEnvironment.PRODUCTION)
    monkeypatch.setattr(metrics.settings.config, "behave_as", BehaveAs.CLOUD)

    assert _sentry_init_options(monkeypatch)["environment"] == "pr-123"


def test_sentry_init_trims_the_environment_override(monkeypatch) -> None:
    monkeypatch.setattr(metrics.settings.config, "sentry_environment", "  pr-123 ")

    assert _sentry_init_options(monkeypatch)["environment"] == "pr-123"


@pytest.mark.parametrize("override", ["", "   "])
def test_sentry_init_computes_the_environment_without_an_override(
    monkeypatch, override
) -> None:
    monkeypatch.setattr(metrics.settings.config, "sentry_environment", override)
    monkeypatch.setattr(metrics.settings.config, "app_env", AppEnvironment.PRODUCTION)
    monkeypatch.setattr(metrics.settings.config, "behave_as", BehaveAs.CLOUD)

    environment = _sentry_init_options(monkeypatch)["environment"]

    assert environment == "app:prod-behave:cloud"


def test_sentry_settings_read_the_deployment_env_vars(monkeypatch) -> None:
    """The names infra sets: SENTRY_RELEASE, baked into the image, and
    SENTRY_ENVIRONMENT, set per preview."""
    monkeypatch.setenv("SENTRY_RELEASE", "0123abcd")
    monkeypatch.setenv("SENTRY_ENVIRONMENT", "pr-123")

    config = Config()

    assert config.sentry_release == "0123abcd"
    assert config.sentry_environment == "pr-123"


def test_sentry_settings_are_empty_for_blank_deployment_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("SENTRY_RELEASE", "")
    monkeypatch.setenv("SENTRY_ENVIRONMENT", "")

    config = Config()

    assert (config.sentry_release, config.sentry_environment) == ("", "")


# ---------- personal data ----------


def _event_with_personal_user() -> dict:
    return {
        "level": "error",
        "user": {
            "id": "user-1",
            "email": "ada@example.com",
            "ip_address": "203.0.113.7",
            "name": "Ada Lovelace",
            "segment": "pro",
        },
    }


def test_before_send_keeps_only_the_user_id() -> None:
    event = _before_send(_event_with_personal_user(), hint={})

    assert event is not None
    assert event["user"] == {"id": "user-1"}


def test_before_send_transaction_keeps_only_the_user_id() -> None:
    event = metrics._before_send_transaction(_event_with_personal_user(), hint={})

    assert event["user"] == {"id": "user-1"}


def test_sentry_init_sends_no_default_pii(monkeypatch) -> None:
    assert _sentry_init_options(monkeypatch)["send_default_pii"] is False


def test_before_send_drops_a_user_left_with_no_id() -> None:
    event = _before_send(
        {"level": "error", "user": {"email": "ada@example.com"}}, hint={}
    )

    assert event is not None
    assert "user" not in event


def test_before_send_transaction_drops_a_user_left_with_no_id() -> None:
    event = metrics._before_send_transaction(
        {"user": {"ip_address": "203.0.113.7"}}, hint={}
    )

    assert "user" not in event


def test_before_send_filters_emails_and_ips_from_extras_and_frame_vars() -> None:
    def notify_user():
        email = "ada@example.com"
        user_email = "ada.lovelace@example.com"
        recipient = "ada@example.org"
        ip = "203.0.113.7"
        ip_address = "198.51.100.4"
        email_count = 1
        if email and user_email and recipient and ip and ip_address and email_count:
            raise RuntimeError("could not send the notification")

    try:
        notify_user()
    except RuntimeError:
        event, hint = event_from_exception(
            sys.exc_info(), client_options=DEFAULT_OPTIONS
        )
    event["extra"] = {
        "Email": "ada@example.com",
        "context": {"user_email": "ada.lovelace@example.com", "ip": "203.0.113.7"},
        "ip_address": "198.51.100.4",
        "email_verified": True,
        "ip_allowlist_enabled": True,
    }

    scrubbed = _before_send(serialize(event), hint)

    assert scrubbed is not None
    body = json.loads(json_dumps(scrubbed))
    assert body["extra"]["Email"] == "[Filtered]"
    assert body["extra"]["context"] == {
        "user_email": "[Filtered]",
        "ip": "[Filtered]",
    }
    assert body["extra"]["ip_address"] == "[Filtered]"
    frame_vars = body["exception"]["values"][0]["stacktrace"]["frames"][-1]["vars"]
    for name in ("email", "user_email", "recipient", "ip", "ip_address"):
        assert frame_vars[name] == "[Filtered]", name
    # Keys that only mention an email or IP stay readable for debugging.
    assert frame_vars["email_count"] == "1"
    assert body["extra"]["email_verified"] is True
    assert body["extra"]["ip_allowlist_enabled"] is True


def test_an_asgi_scopes_credential_headers_are_masked_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frame's ASGI scope holds the headers as pairs of bytes, which Sentry
    shows by their reprs; a header named as a credential is masked whatever
    its value, which the request's own credentials needn't include."""
    monkeypatch.setattr(metrics, "request_credentials", lambda: ())

    def handle() -> None:
        scope = {
            "type": "http",
            "headers": [
                (b"cookie", b"__cf_bm=opaque-cookie-Zq81mN2p"),
                (b"x-api-key", b"opaque-api-key-Zq81mN2p"),
                (b"accept", b"*/*"),
            ],
        }
        if scope:
            raise RuntimeError("the handler failed")

    try:
        handle()
    except RuntimeError:
        event, hint = event_from_exception(
            sys.exc_info(), client_options=DEFAULT_OPTIONS
        )

    scrubbed = _before_send(serialize(event), hint)

    assert scrubbed is not None
    frame_vars = scrubbed["exception"]["values"][0]["stacktrace"]["frames"][-1]["vars"]
    headers = frame_vars["scope"]["headers"]
    assert [name for name, _ in headers] == ["b'cookie'", "b'x-api-key'", "b'accept'"]
    assert headers[0][1] == headers[1][1] == "[Filtered]"
    assert headers[2][1] == "b'*/*'"


@pytest.mark.parametrize(
    "name", ["b'cookie'", "b'set-cookie'", "b'authorization'", "b'x-api-key'"]
)
def test_mask_headers_masks_a_byte_form_credential_header(name: str) -> None:
    """A header name as Sentry shows an ASGI scope's bytes is read whole, so in
    either shape `_mask_headers` reads, it keeps no part of its value."""
    pairs = [[name, "b'opaque-value'"], ["b'accept'", "b'*/*'"]]
    mapping = {name: "opaque-value", "b'accept'": "*/*"}

    metrics._mask_headers("headers", pairs)
    metrics._mask_headers("headers", mapping)

    assert pairs == [[name, "[Filtered]"], ["b'accept'", "b'*/*'"]]
    assert mapping == {name: "[Filtered]", "b'accept'": "*/*"}


def _event_with_ip_headers() -> dict:
    return {
        "level": "error",
        "request": {
            "url": "https://backend.agpt.co/api/library",
            "headers": {
                "User-Agent": "ua",
                "Content-Type": "application/json",
                "Cf-Connecting-Ip": "203.0.113.7",
                "True-Client-Ip": "203.0.113.7",
                "Forwarded": "for=203.0.113.7",
                "X-Forwarded-For": "203.0.113.7, 10.0.0.1",
                "X-Real-Ip": "203.0.113.7",
                "Cf-Ipcountry": "PL",
                "Cf-Ipcity": "Warsaw",
                "Cf-Iplatitude": "52.2",
                "X-Vercel-Ip-City": "Warsaw",
                # The frontend's signed client-IP token carries the visitor's IP too.
                CLIENT_IP_TOKEN_HEADER: "203.0.113.7",
                "Cf-Ray": "8a1b2c3d4e5f6789-WAW",
            },
        },
    }


_KEPT_HEADERS = {
    "User-Agent": "ua",
    "Content-Type": "application/json",
    "Cf-Ray": "8a1b2c3d4e5f6789-WAW",
}


def test_before_send_drops_the_visitors_ip_and_location_headers() -> None:
    event = _before_send(_event_with_ip_headers(), hint={})

    assert event is not None
    assert event["request"]["headers"] == _KEPT_HEADERS


def test_before_send_transaction_drops_the_visitors_ip_and_location_headers() -> None:
    event = metrics._before_send_transaction(_event_with_ip_headers(), hint={})

    assert event["request"]["headers"] == _KEPT_HEADERS
