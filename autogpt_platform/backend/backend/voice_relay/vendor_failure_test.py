"""Why a Gemini Live connect or session failed: read from the HTTP status, the
API status and the close details a failure carries, never from the free text
of a message, and logged with no key, token or anything said on the call."""

import pytest
from google.genai import errors
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError, InvalidStatus
from websockets.frames import Close
from websockets.http11 import Response

from backend.voice_relay.vendor_failure import (
    MAX_REASON_CHARS,
    REDACTED,
    VendorFailure,
    scrubbed,
    vendor_failure,
)

API_KEY = "AIzaSyD-not-a-real-key-0123456789abcdefg"
SAID = "I run a bakery with five staff"


def refused(status: int) -> InvalidStatus:
    """The server refusing the Live socket's handshake with *status*."""
    return InvalidStatus(Response(status, "Refused", Headers()))


def api_error(code: int, status: str) -> errors.APIError:
    """An error the server sent as JSON, as the SDK raises it."""
    body = {"error": {"code": code, "message": "No.", "status": status}}
    kind = errors.ClientError if code < 500 else errors.ServerError
    return kind(code, body)


def closed(code: int, reason: str) -> errors.APIError:
    """A socket the server closed during setup or mid-session, as the SDK
    raises it."""
    return errors.APIError(code, reason)


@pytest.mark.parametrize(
    "status,reason",
    [
        (429, "quota"),
        (402, "payment"),
        (403, "forbidden"),
        (503, "unavailable"),
        (500, "other"),
        (401, "other"),
    ],
)
def test_a_refused_connection_reads_its_http_status(status, reason):
    failure = vendor_failure(refused(status))

    assert (failure.reason, failure.http_status) == (reason, status)
    assert failure.error_type == "InvalidStatus"
    assert f"HTTP {status}" in failure.describe()


@pytest.mark.parametrize(
    "code,status,reason",
    [
        (429, "RESOURCE_EXHAUSTED", "quota"),
        (400, "RESOURCE_EXHAUSTED", "quota"),
        (402, "FAILED_PRECONDITION", "payment"),
        (403, "PERMISSION_DENIED", "forbidden"),
        (503, "UNAVAILABLE", "unavailable"),
        (500, "INTERNAL", "other"),
    ],
)
def test_an_error_the_server_sent_reads_its_code_and_status(code, status, reason):
    failure = vendor_failure(api_error(code, status))

    assert (failure.reason, failure.http_status, failure.api_status) == (
        reason,
        code,
        status,
    )
    assert failure.close_code is None


def test_a_socket_closed_by_the_server_reads_its_close_code_and_reason():
    failure = vendor_failure(closed(1011, "Internal error encountered."))

    assert (failure.reason, failure.close_code, failure.close_reason) == (
        "other",
        1011,
        "Internal error encountered.",
    )
    assert failure.http_status is None
    assert failure.describe() == (
        "other (APIError, close 1011, reason: Internal error encountered.)"
    )


@pytest.mark.parametrize(
    "words",
    [
        "You exceeded your current quota, please check your plan and billing.",
        "RESOURCE_EXHAUSTED",
        "429 Too Many Requests",
        "Payment required: 402",
    ],
)
def test_a_close_reason_s_words_never_decide_why(words):
    """Only a status reads as a reason; a close reason is free text."""
    assert vendor_failure(closed(1008, words)).reason == "other"


def test_a_send_on_a_closed_socket_reads_its_close_details():
    failure = vendor_failure(
        ConnectionClosedError(Close(1011, "Internal error."), None)
    )

    assert (failure.close_code, failure.close_reason) == (1011, "Internal error.")


def test_a_socket_that_dropped_without_a_close_reads_no_details():
    failure = vendor_failure(ConnectionClosedError(None, None))

    assert failure == VendorFailure(error_type="ConnectionClosedError")


def test_a_failure_the_vendor_explained_nothing_about_is_other():
    failure = vendor_failure(TimeoutError())

    assert failure.describe() == "other (TimeoutError)"
    assert failure.summary() == "other (TimeoutError)"


@pytest.mark.parametrize(
    "error",
    [ConnectionResetError(), closed(1011, "Internal error."), closed(1001, "")],
)
def test_a_drop_after_the_closing_notice_is_a_server_reset(error):
    assert vendor_failure(error, after_notice=True).reason == "server_reset"
    assert vendor_failure(error).reason == "other"


def test_a_status_says_more_than_the_closing_notice():
    assert vendor_failure(refused(429), after_notice=True).reason == "quota"


def test_an_api_status_that_is_not_a_status_name_is_left_out():
    error = errors.APIError(500, {"error": {"code": 500, "status": SAID}})

    failure = vendor_failure(error)

    assert failure.api_status is None
    assert SAID not in failure.describe()


def test_a_close_reason_carries_no_key_token_credential_or_quoted_words():
    reason = (
        f"Rejected: key {API_KEY} at wss://user:hunter2@example.com/ws?key={API_KEY}"
        f"&alt=json, Authorization: Bearer ya29.secret-token, "
        f"content \"{SAID}\" and '{SAID}'"
    )

    described = vendor_failure(closed(1007, reason)).describe()

    for secret in (API_KEY, "hunter2", "ya29.secret-token", SAID):
        assert secret not in described
    assert REDACTED in described
    assert "close 1007" in described


def test_a_long_close_reason_is_cut():
    assert len(scrubbed("x" * 1_000) or "") == MAX_REASON_CHARS


@pytest.mark.parametrize("text", [None, "", "   ", "\n"])
def test_nothing_to_say_reads_as_none(text):
    assert scrubbed(text) is None
