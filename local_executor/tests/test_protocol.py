"""Tests for protocol parsing and message envelopes."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from autogpt_local_executor.protocol import (
    VERSION,
    AttachSessionMessage,
    CommandResultMessage,
    ErrorCode,
    ExecuteCommandMessage,
    FileReadMessage,
    HelloAckMessage,
    HelloMessage,
    HelloPayload,
    MessageType,
    Platform,
    ProtocolVersionMismatch,
    Shell,
    dump_message,
    make_ack,
    make_error,
    make_pong,
    negotiate_version,
    new_id,
    parse_message,
)


def test_parse_ping() -> None:
    msg = parse_message('{"type": "PING", "id": "abc", "ts": 1.0, "payload": {}}')
    assert msg.type == MessageType.PING
    assert msg.id == "abc"


def test_parse_hello_ack() -> None:
    raw = json.dumps(
        {
            "type": "HELLO_ACK",
            "id": "abc",
            "ts": 1.0,
            "payload": {
                "session_id": "s1",
                "granted_capabilities": ["shell", "files"],
                "max_file_size_bytes": 1024,
                "command_timeout_seconds": 5,
                "max_concurrent": 2,
            },
        }
    )
    msg = parse_message(raw)
    assert isinstance(msg, HelloAckMessage)
    assert msg.payload.session_id == "s1"
    assert msg.payload.granted_capabilities == ["shell", "files"]


def test_parse_control_directory_and_attach_messages() -> None:
    listing = parse_message(
        json.dumps(
            {
                "type": "DIRECTORY_LIST_REQUEST",
                "id": "list",
                "ts": 1.0,
                "payload": {"browse_id": None, "directory_ref": None},
            }
        )
    )
    assert listing.type == MessageType.DIRECTORY_LIST_REQUEST

    attach = parse_message(
        json.dumps(
            {
                "type": "ATTACH_SESSION",
                "id": "attach",
                "ts": 1.0,
                "payload": {
                    "session_id": "session-1",
                    "browse_id": "browse-1",
                    "directory_ref": "ref-1",
                    "expected_connection_id": "connection-1",
                },
            }
        )
    )
    assert isinstance(attach, AttachSessionMessage)
    assert attach.payload.expected_connection_id == "connection-1"


def test_directory_list_request_rejects_partial_reference_pair() -> None:
    with pytest.raises(ValidationError, match="both be null or both be set"):
        parse_message(
            json.dumps(
                {
                    "type": "DIRECTORY_LIST_REQUEST",
                    "id": "list",
                    "ts": 1.0,
                    "payload": {"browse_id": "browse-1", "directory_ref": None},
                }
            )
        )


def test_directory_list_cursor_requires_reference_pair() -> None:
    with pytest.raises(ValidationError, match="cursor requires"):
        parse_message(
            json.dumps(
                {
                    "type": "DIRECTORY_LIST_REQUEST",
                    "id": "list",
                    "ts": 1.0,
                    "payload": {
                        "browse_id": None,
                        "directory_ref": None,
                        "cursor": "next-page",
                    },
                }
            )
        )


def test_attach_rejects_session_id_that_can_escape_url_path() -> None:
    with pytest.raises(ValidationError):
        parse_message(
            json.dumps(
                {
                    "type": "ATTACH_SESSION",
                    "id": "attach",
                    "ts": 1.0,
                    "payload": {
                        "session_id": "../control?admin=true",
                        "browse_id": "browse-1",
                        "directory_ref": "ref-1",
                        "expected_connection_id": None,
                    },
                }
            )
        )


def test_parse_execute_command_with_argv() -> None:
    raw = json.dumps(
        {
            "type": "EXECUTE_COMMAND",
            "id": "abc",
            "ts": 1.0,
            "payload": {
                "argv": ["ls", "-la"],
                "command": None,
                "shell": "auto",
                "cwd": "/tmp",
                "timeout_seconds": 5,
                "env": {},
            },
        }
    )
    msg = parse_message(raw)
    assert isinstance(msg, ExecuteCommandMessage)
    assert msg.payload.argv == ["ls", "-la"]
    assert msg.payload.command is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_file_size_bytes", 0),
        ("max_file_size_bytes", -1),
        ("command_timeout_seconds", 0),
        ("command_timeout_seconds", -1),
        ("max_concurrent", 0),
        ("max_concurrent", -1),
    ],
)
def test_parse_hello_ack_rejects_nonpositive_limits(field: str, value: int) -> None:
    payload = {
        "session_id": "s1",
        "granted_capabilities": ["files"],
        "max_file_size_bytes": 1024,
        "command_timeout_seconds": 5,
        "max_concurrent": 2,
    }
    payload[field] = value
    raw = json.dumps(
        {
            "type": "HELLO_ACK",
            "id": "abc",
            "ts": 1.0,
            "payload": payload,
        }
    )

    with pytest.raises(ValidationError):
        parse_message(raw)


@pytest.mark.parametrize("timeout", [0, -1])
def test_parse_execute_command_rejects_nonpositive_timeout(timeout: int) -> None:
    raw = json.dumps(
        {
            "type": "EXECUTE_COMMAND",
            "id": "abc",
            "ts": 1.0,
            "payload": {"argv": ["echo"], "timeout_seconds": timeout},
        }
    )

    with pytest.raises(ValidationError):
        parse_message(raw)


def test_command_result_round_trips_output_truncated() -> None:
    raw = json.dumps(
        {
            "type": "COMMAND_RESULT",
            "id": "abc",
            "ts": 1.0,
            "payload": {
                "stdout": "partial",
                "stderr": "",
                "exit_code": 0,
                "timed_out": False,
                "duration_seconds": 0.1,
                "output_truncated": True,
            },
        }
    )

    msg = parse_message(raw)

    assert isinstance(msg, CommandResultMessage)
    assert msg.payload.output_truncated is True


def test_parse_file_read_format_text_default() -> None:
    raw = json.dumps(
        {
            "type": "FILE_READ",
            "id": "abc",
            "ts": 1.0,
            "payload": {"path": "/tmp/a.txt"},
        }
    )
    msg = parse_message(raw)
    assert isinstance(msg, FileReadMessage)
    assert msg.payload.format == "text"
    assert msg.payload.encoding == "utf-8"


def test_parse_rejects_unknown_type() -> None:
    raw = json.dumps({"type": "BOGUS", "id": "x", "ts": 1.0, "payload": {}})
    with pytest.raises(ValidationError):
        parse_message(raw)


def _computer_call_frame(depth: int) -> str:
    arguments = '{"a":' * depth + "1" + "}" * depth
    return (
        '{"type":"COMPUTER_TOOL_CALL","id":"req","ts":1.0,'
        '"payload":{"tool":"click","arguments":' + arguments + "}}"
    )


@pytest.mark.parametrize("depth", [600, 200_000], ids=["deep", "past-the-json-parser"])
def test_a_frame_nested_too_deeply_is_refused(depth: int) -> None:
    """Arguments nested some hundreds of levels deep parsed, and a computer
    tool call carrying them reached cua-driver while its audit record failed:
    the audit writer, its canonicalizer and the argument redaction all
    recurse. Past the JSON parser's own limit, RecursionError escaped."""
    with pytest.raises(ValueError, match="nested"):
        parse_message(_computer_call_frame(depth))


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
def test_a_frame_with_a_number_json_cannot_hold_is_refused(number: str) -> None:
    """Python's json reads NaN and Infinity, which JSON has no spelling
    for, and reads 1e999 as infinity. A computer tool call carrying one
    reached cua-driver while its audit record failed, since the canonical
    form refuses them."""
    frame = (
        '{"type":"COMPUTER_TOOL_CALL","id":"req","ts":1.0,'
        '"payload":{"tool":"click","arguments":{"steps":[{"x":' + number + "}]}}}"
    )

    with pytest.raises(ValueError, match="finite"):
        parse_message(frame)


def test_a_frame_with_ordinary_numbers_parses() -> None:
    frame = (
        '{"type":"COMPUTER_TOOL_CALL","id":"req","ts":1.5e9,'
        '"payload":{"tool":"click","arguments":{"x":-12.5,"y":3,"z":1e300}}}'
    )

    assert parse_message(frame).payload.arguments == {"x": -12.5, "y": 3, "z": 1e300}


# PROTOCOL.md: a frame whose objects and arrays nest more than 64 levels
# deep is dropped. The number as documented, not the constant, so a change
# to either shows.
_DOCUMENTED_NESTING_LIMIT = 64


def _computer_call_frame_of_arrays(depth: int) -> str:
    """Like _computer_call_frame, with the arguments object holding arrays
    nested to `depth` levels in all."""
    arrays = "[" * (depth - 1) + "1" + "]" * (depth - 1)
    return (
        '{"type":"COMPUTER_TOOL_CALL","id":"req","ts":1.0,'
        '"payload":{"tool":"click","arguments":{"a":' + arrays + "}}}"
    )


@pytest.mark.parametrize(
    "frame", [_computer_call_frame, _computer_call_frame_of_arrays], ids=["objects", "arrays"]
)
def test_a_frame_nested_exactly_to_the_limit_parses(frame) -> None:
    # The envelope and the payload add two levels to the arguments' own.
    message = parse_message(frame(_DOCUMENTED_NESTING_LIMIT - 2))

    assert message.payload.tool == "click"


@pytest.mark.parametrize(
    "frame", [_computer_call_frame, _computer_call_frame_of_arrays], ids=["objects", "arrays"]
)
def test_a_frame_nested_one_level_past_the_limit_is_refused(frame) -> None:
    """The deep cases start hundreds of levels down, so the limit could move
    by one, or be off by one, with every test passing."""
    with pytest.raises(ValueError, match="nested"):
        parse_message(frame(_DOCUMENTED_NESTING_LIMIT - 1))


def test_parse_rejects_missing_id() -> None:
    raw = json.dumps({"type": "PING", "ts": 1.0, "payload": {}})
    with pytest.raises(ValidationError):
        parse_message(raw)


def test_parse_rejects_extra_payload_field() -> None:
    raw = json.dumps(
        {
            "type": "PING",
            "id": "x",
            "ts": 1.0,
            "payload": {"unexpected": "value"},
        }
    )
    with pytest.raises(ValidationError):
        parse_message(raw)


def test_dump_round_trip_hello() -> None:
    msg = HelloMessage(
        id=new_id(),
        ts=1.0,
        payload=HelloPayload(
            shim_version="0.0.1",
            machine_id="m1",
            platform=Platform.LINUX,
            arch="x86_64",  # type: ignore[arg-type]
            capabilities=["shell", "files"],
            allowed_root="/home/u/ws",
        ),
    )
    raw = dump_message(msg)
    again = parse_message(raw)
    assert isinstance(again, HelloMessage)
    assert again.payload.machine_id == "m1"


def test_make_error_uses_code_enum() -> None:
    err = make_error("x", ErrorCode.PATH_OUTSIDE_ALLOWED_ROOT, "oops")
    payload = json.loads(dump_message(err))["payload"]
    assert payload["code"] == "PATH_OUTSIDE_ALLOWED_ROOT"


def test_make_ack_round_trip() -> None:
    ack = make_ack("x")
    raw = dump_message(ack)
    again = parse_message(raw)
    assert again.type == MessageType.ACK
    assert again.payload.ok is True


def test_make_pong_round_trip() -> None:
    pong = make_pong("x")
    raw = dump_message(pong)
    again = parse_message(raw)
    assert again.type == MessageType.PONG


def test_shell_enum_accepts_auto() -> None:
    raw = json.dumps(
        {
            "type": "EXECUTE_COMMAND",
            "id": "x",
            "ts": 1.0,
            "payload": {"command": "echo hi", "argv": None},
        }
    )
    msg = parse_message(raw)
    assert isinstance(msg, ExecuteCommandMessage)
    assert msg.payload.shell == Shell.AUTO


# ── Wire-protocol version (#35) ──────────────────────────────────────────────


def test_envelope_emits_version_field_by_default() -> None:
    """Every envelope serializes with the current VERSION."""
    ack = make_ack("x")
    payload = json.loads(dump_message(ack))
    assert payload["version"] == VERSION


def test_hello_payload_includes_protocol_version() -> None:
    """HELLO's payload carries the shim's max-supported protocol version."""
    msg = HelloMessage(
        id=new_id(),
        ts=1.0,
        payload=HelloPayload(
            shim_version="0.0.1",
            machine_id="m1",
            platform=Platform.LINUX,
            arch="x86_64",  # type: ignore[arg-type]
            capabilities=["shell", "files"],
            allowed_root="/home/u/ws",
        ),
    )
    payload = json.loads(dump_message(msg))
    assert payload["payload"]["protocol_version"] == VERSION


def test_parse_envelope_tolerates_missing_version() -> None:
    """Receivers must be lenient: an inbound frame without `version` still parses."""
    raw = json.dumps({"type": "PING", "id": "abc", "ts": 1.0, "payload": {}})
    msg = parse_message(raw)
    # The default kicks in.
    assert msg.version == VERSION


def test_parse_envelope_accepts_explicit_version() -> None:
    raw = json.dumps({"type": "PING", "id": "abc", "ts": 1.0, "version": "1.4", "payload": {}})
    msg = parse_message(raw)
    assert msg.version == "1.4"


def test_negotiate_version_same_version() -> None:
    assert negotiate_version("1.0", "1.0") == "1.0"


def test_negotiate_version_minor_floor_shim_lower() -> None:
    assert negotiate_version("1.2", "1.7") == "1.2"


def test_negotiate_version_minor_floor_platform_lower() -> None:
    assert negotiate_version("1.7", "1.2") == "1.2"


def test_negotiate_version_major_mismatch_shim_older() -> None:
    with pytest.raises(ProtocolVersionMismatch) as exc_info:
        negotiate_version("1.5", "2.0")
    err = exc_info.value
    assert err.shim_max == "1.5"
    assert err.platform_max == "2.0"
    reason = err.to_close_reason()
    assert reason["error"] == "PROTOCOL_VERSION_MISMATCH"
    assert reason["shim_max"] == "1.5"
    assert reason["platform_max"] == "2.0"
    assert "hint" not in reason


def test_negotiate_version_major_mismatch_platform_older() -> None:
    with pytest.raises(ProtocolVersionMismatch):
        negotiate_version("2.0", "1.5")


def test_negotiate_version_malformed_input() -> None:
    with pytest.raises(ValueError):
        negotiate_version("1", "1.0")
    with pytest.raises(ValueError):
        negotiate_version("v1.0", "1.0")
    with pytest.raises(ValueError):
        negotiate_version("1.0.1", "1.0")


def test_parse_hello_ack_with_protocol_version() -> None:
    raw = json.dumps(
        {
            "type": "HELLO_ACK",
            "id": "abc",
            "ts": 1.0,
            "version": "1.0",
            "payload": {
                "session_id": "s1",
                "granted_capabilities": ["shell"],
                "max_file_size_bytes": 1024,
                "command_timeout_seconds": 5,
                "max_concurrent": 2,
                "protocol_version": "1.0",
            },
        }
    )
    msg = parse_message(raw)
    assert isinstance(msg, HelloAckMessage)
    assert msg.payload.protocol_version == "1.0"


def test_parse_hello_ack_defaults_protocol_version() -> None:
    """Receivers stay lenient: a HELLO_ACK without protocol_version still parses
    and defaults to the current VERSION."""
    raw = json.dumps(
        {
            "type": "HELLO_ACK",
            "id": "abc",
            "ts": 1.0,
            "payload": {
                "session_id": "s1",
                "granted_capabilities": ["shell"],
                "max_file_size_bytes": 1024,
                "command_timeout_seconds": 5,
                "max_concurrent": 2,
            },
        }
    )
    msg = parse_message(raw)
    assert isinstance(msg, HelloAckMessage)
    assert msg.payload.protocol_version == VERSION


@pytest.mark.parametrize("version", ["1.-1", "1.+2", "1.\u0661", " 1.1", "1.1 ", "1_0.1"])
def test_a_version_component_must_be_ascii_digits(version: str) -> None:
    """int() accepts signs, spaces, underscores and non-ASCII digits, so
    `1.-1` used to negotiate a negative minor."""
    with pytest.raises(ValueError):
        negotiate_version("1.1", version)
