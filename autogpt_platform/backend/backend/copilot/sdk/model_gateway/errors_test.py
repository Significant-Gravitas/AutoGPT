import pytest

from backend.copilot.sdk.model_gateway.errors import (
    REDACTED,
    error_for_status,
    error_from_payload,
    parse_upstream_error,
    redact,
    retry_headers,
    status_for_type,
)


@pytest.mark.parametrize(
    "body, message, error_type, code",
    [
        (
            b'{"type":"error","error":{"type":"overloaded_error","message":"busy"}}',
            "busy",
            "overloaded_error",
            None,
        ),
        (
            b'{"error":{"message":"nope","type":"invalid_request_error",'
            b'"code":"bad_thing"}}',
            "nope",
            "invalid_request_error",
            "bad_thing",
        ),
        (b'{"error":"plain string error"}', "plain string error", None, None),
        (b'{"detail":"Unauthorized"}', "Unauthorized", None, None),
        (b'{"object":"error","message":"m","code":400}', "m", None, "400"),
        (b"<html>502 Bad Gateway</html>", "<html>502 Bad Gateway</html>", None, None),
    ],
)
def test_error_bodies_are_read_in_every_dialect(body, message, error_type, code):
    error = parse_upstream_error(body)
    assert (error.message, error.error_type, error.code) == (message, error_type, code)


def test_the_error_object_keeps_its_extra_fields():
    error = error_from_payload(
        {"error": {"message": "m", "n_prompt_tokens": 9, "n_ctx": 8}}
    )
    assert error is not None
    assert error.fields["n_prompt_tokens"] == 9


def test_a_payload_that_is_not_an_object_has_no_error():
    assert error_from_payload(["not", "an", "object"]) is None


@pytest.mark.parametrize(
    "status, error_type",
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (413, "request_too_large"),
        (429, "rate_limit_error"),
        (500, "api_error"),
        (503, "api_error"),
        (529, "overloaded_error"),
    ],
)
def test_statuses_map_to_anthropic_error_types(status, error_type):
    error = error_for_status(status, "x")
    assert (error.status, error.error_type) == (status, error_type)
    assert error.body() == {
        "type": "error",
        "error": {"type": error_type, "message": "x"},
    }


def test_a_status_outside_the_error_range_is_a_bad_gateway():
    error = error_for_status(302, "")
    assert (error.status, error.error_type) == (502, "api_error")
    assert "302" in error.message


def test_an_upstream_type_anthropic_knows_is_kept():
    assert error_for_status(500, "x", error_type="overloaded_error").error_type == (
        "overloaded_error"
    )
    assert (
        error_for_status(500, "x", error_type="server_error").error_type == "api_error"
    )


def test_error_types_map_back_to_statuses():
    assert status_for_type("rate_limit_error") == 429
    assert status_for_type("overloaded_error") == 529
    assert status_for_type("something_else") == 500
    assert status_for_type(None) == 500


@pytest.mark.parametrize(
    "text, leaked",
    [
        ("key sk-proj-abcdefghijklmnop rejected", "sk-proj-abcdefghijklmnop"),
        ("Authorization: Bearer abc.def.ghi-123", "abc.def.ghi-123"),
        ('{"api_key": "hunter2-secret"}', "hunter2-secret"),
        ("access_token=tok_123456789", "tok_123456789"),
        (
            "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJl here",
            "eyJhbGciOiJIUzI1NiJ9",
        ),
    ],
)
def test_redaction_removes_anything_shaped_like_a_credential(text, leaked):
    redacted = redact(text)
    assert leaked not in redacted
    assert REDACTED in redacted


def test_redaction_removes_the_route_secret_wherever_it_appears():
    redacted = redact(
        "echo: my-upstream-key-42 / my-upstream-key-42", ["my-upstream-key-42"]
    )
    assert "my-upstream-key-42" not in redacted


def test_redaction_bounds_the_message_and_collapses_whitespace():
    redacted = redact("word\n\n " * 1000)
    assert len(redacted) <= 1000
    assert "\n" not in redacted


def test_only_the_retry_hints_the_cli_reads_are_passed_on():
    hints = retry_headers(
        {
            "Retry-After": " 12 ",
            "retry-after-ms": "1500",
            "x-ratelimit-remaining": "0",
            "Set-Cookie": "session=abc",
        }
    )
    assert hints == {"retry-after": "12", "retry-after-ms": "1500"}


def test_a_malformed_retry_hint_is_dropped():
    assert retry_headers({"Retry-After": "soon; also evil\r\nX: y"}) == {}
