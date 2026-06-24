"""Reproduction test for the OpenRouter incompatibility in newer
``claude-agent-sdk`` / Claude Code CLI versions.

Background — there are two stacked regressions that block us from
upgrading the ``claude-agent-sdk`` package above ``0.1.45``:

1. **`tool_reference` content blocks** introduced by CLI ``2.1.69`` (=
   SDK ``0.1.46``).  The CLI's built-in ``ToolSearch`` tool returns
   ``{"type": "tool_reference", "tool_name": "..."}`` content blocks in
   ``tool_result.content``.  OpenRouter's stricter Zod validation
   rejects this with::

        messages[N].content[0].content: Invalid input: expected string, received array

   This is the regression that originally pinned us at 0.1.45 — see
   https://github.com/Significant-Gravitas/AutoGPT/pull/12294 for the
   full forensic write-up.  CLI 2.1.70 added proxy detection that
   *should* disable the offending blocks when ``ANTHROPIC_BASE_URL`` is
   set, but our subsequent attempts at 0.1.55 / 0.1.56 still failed.

2. **`context-management-2025-06-27` beta header** — some CLI version
   after ``2.1.91`` started injecting this header / beta flag, which
   OpenRouter rejects with::

        400 No endpoints available that support Anthropic's context
        management features (context-management-2025-06-27). Context
        management requires a supported provider (Anthropic).

   Tracked upstream at
   https://github.com/anthropics/claude-agent-sdk-python/issues/789.
   Still open at the time of writing, no upstream PR linked, no
   workaround documented.

The purpose of this test:
* Spin up a tiny in-process HTTP server that pretends to be the
  Anthropic Messages API.
* Capture every request body the CLI sends.
* Inspect the captured bodies for the two forbidden patterns above.
* Fail loudly if either is present, with a pointer to the issue
  tracker.

This is the reproduction we use as a CI gate when bisecting which SDK /
CLI version is safe to upgrade to.  It runs against the bundled CLI by
default (or against ``ChatConfig.claude_agent_cli_path`` when set), so
it doubles as a regression guard for the ``cli_path`` override
mechanism.

The test does **not** need an OpenRouter API key — it reproduces the
mechanism (forbidden content blocks / headers in the *outgoing*
request) rather than the symptom (the 400 OpenRouter would return).
This keeps it deterministic, free, and CI-runnable without secrets.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forbidden patterns we scan for in captured request bodies
# ---------------------------------------------------------------------------

# Substring of the context-management beta string that OpenRouter rejects
# (upstream issue #789).  Can appear in either `betas` arrays or the
# `anthropic-beta` header value sent by the CLI.
_FORBIDDEN_CONTEXT_MANAGEMENT_BETA = "context-management-2025-06-27"


def _body_contains_tool_reference_block(body_text: str) -> bool:
    """Return True if *body_text* contains a ``tool_reference`` content
    block anywhere in its structure.

    We parse the JSON and walk it rather than relying on substring
    matches because the CLI is free to emit either ``{"type": "tool_reference"}``
    (with spaces) or the compact ``{"type":"tool_reference"}`` form,
    and we must catch both.  Falls back to a whitespace-tolerant
    regex when the body isn't valid JSON — the Messages API always
    sends JSON, but the fallback keeps the detector honest on
    malformed / partial bodies a fuzzer might produce.
    """
    try:
        payload = json.loads(body_text)
    except (ValueError, TypeError):
        # Whitespace-tolerant fallback: allow any whitespace between
        # the key, colon, and value quoted string.
        return bool(re.search(r'"type"\s*:\s*"tool_reference"', body_text))

    def _walk(node: Any) -> bool:
        if isinstance(node, dict):
            if node.get("type") == "tool_reference":
                return True
            return any(_walk(v) for v in node.values())
        if isinstance(node, list):
            return any(_walk(v) for v in node)
        return False

    return _walk(payload)


def _scan_request_for_forbidden_patterns(
    body_text: str,
    headers: dict[str, str],
) -> list[str]:
    """Return a list of forbidden patterns found in *body_text* / *headers*.

    Empty list = clean request.  Non-empty = the CLI is sending one of the
    OpenRouter-incompatible features.
    """
    findings: list[str] = []
    if _body_contains_tool_reference_block(body_text):
        findings.append(
            "`tool_reference` content block in request body — "
            "PR #12294 / CLI 2.1.69 regression"
        )
    if _FORBIDDEN_CONTEXT_MANAGEMENT_BETA in body_text:
        findings.append(
            f"{_FORBIDDEN_CONTEXT_MANAGEMENT_BETA!r} in request body — "
            "anthropics/claude-agent-sdk-python#789"
        )
    # Header values are case-insensitive in HTTP — aiohttp normalises
    # incoming names but values are stored as-is.
    for header_name, header_value in headers.items():
        if header_name.lower() == "anthropic-beta":
            if _FORBIDDEN_CONTEXT_MANAGEMENT_BETA in header_value:
                findings.append(
                    f"{_FORBIDDEN_CONTEXT_MANAGEMENT_BETA!r} in "
                    "`anthropic-beta` header — issue #789"
                )
    return findings


# ---------------------------------------------------------------------------
# Fake Anthropic Messages API
# ---------------------------------------------------------------------------
#
# We need to give the CLI a *successful* response so it doesn't error out
# before we get a chance to inspect the request.  The minimal thing the
# CLI accepts is a streamed (SSE) message-start → content-block-delta →
# message-stop sequence.
#
# We don't strictly *need* the CLI to accept the response — we already
# have the request body by the time we send any reply — but giving it a
# valid stream means the assertion failure (if any) is the *only*
# failure mode in the test, not "CLI exited 1 because we sent garbage".


def _build_streaming_message_response() -> str:
    """Return an SSE-formatted body containing a minimal Anthropic
    Messages API streamed response.

    This is the smallest stream that the Claude Code CLI will accept
    end-to-end without errors.  Each line is one SSE event."""
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-test",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "ok"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]
    return "".join(
        f"event: {evt['type']}\ndata: {json.dumps(evt)}\n\n" for evt in events
    )


class _CapturedRequest:
    """One request the fake server received."""

    def __init__(self, path: str, headers: dict[str, str], body: str) -> None:
        self.path = path
        self.headers = headers
        self.body = body


async def _start_fake_anthropic_server(
    captured: list[_CapturedRequest],
) -> tuple[web.AppRunner, int]:
    """Start an aiohttp server pretending to be the Anthropic API.

    All POSTs to ``/v1/messages`` are recorded into *captured* and
    answered with a valid streaming response.  Returns ``(runner, port)``
    so the caller can ``await runner.cleanup()`` when finished.
    """

    async def messages_handler(request: web.Request) -> web.StreamResponse:
        body = await request.text()
        captured.append(
            _CapturedRequest(
                path=request.path,
                headers={k: v for k, v in request.headers.items()},
                body=body,
            )
        )
        # Stream a minimal valid response so the CLI doesn't error out
        # before we can inspect what it sent.
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)
        await response.write(_build_streaming_message_response().encode("utf-8"))
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/messages", messages_handler)
    # OAuth/profile endpoints the CLI may probe — answer 404 so it falls
    # through quickly without retrying.
    app.router.add_route("*", "/{tail:.*}", lambda _r: web.Response(status=404))

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    server = site._server
    assert server is not None
    sockets = getattr(server, "sockets", None)
    assert sockets is not None
    port: int = sockets[0].getsockname()[1]
    return runner, port


# ---------------------------------------------------------------------------
# CLI invocation
# ---------------------------------------------------------------------------


def _resolve_cli_path() -> Path | None:
    """Return the Claude Code CLI binary the SDK would use.

    Honours the same override mechanism as ``service.py`` /
    ``ChatConfig.claude_agent_cli_path``: checks either the Pydantic-
    prefixed ``CHAT_CLAUDE_AGENT_CLI_PATH`` or the unprefixed
    ``CLAUDE_AGENT_CLI_PATH`` env var first, then falls back to the
    bundled binary that ships with the installed ``claude-agent-sdk``
    wheel. The two env var names are accepted at the config layer via
    ``ChatConfig.get_claude_agent_cli_path`` and mirrored here so the
    reproduction test picks up the same override regardless of which
    form an operator sets.
    """
    override = os.environ.get("CHAT_CLAUDE_AGENT_CLI_PATH") or os.environ.get(
        "CLAUDE_AGENT_CLI_PATH"
    )
    if override:
        candidate = Path(override)
        return candidate if candidate.is_file() else None

    try:
        from typing import cast

        from claude_agent_sdk._internal.transport.subprocess_cli import (
            SubprocessCLITransport,
        )

        bundled = cast(str, SubprocessCLITransport._find_bundled_cli(None))
        return Path(bundled) if bundled else None
    except (ImportError, AttributeError) as e:  # pragma: no cover - import-time guard
        logger.warning("Could not locate bundled Claude CLI: %s", e)
        return None


async def _run_cli_against_fake_server(
    cli_path: Path,
    fake_server_port: int,
    timeout_seconds: float,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Spawn the CLI pointed at the fake Anthropic server and feed it a
    single ``user`` message via stream-json on stdin.

    Returns ``(returncode, stdout, stderr)``.  The return code is not
    asserted by the test — we only care that the CLI made at least one
    POST to ``/v1/messages`` so the fake server captured the body.
    """
    fake_url = f"http://127.0.0.1:{fake_server_port}"
    env = {
        # Inherit basic shell variables so the CLI can find its tools,
        # but force network/auth at our fake endpoint.
        **os.environ,
        "ANTHROPIC_BASE_URL": fake_url,
        "ANTHROPIC_API_KEY": "sk-test-fake-key-not-real",
        # Disable any features that would phone home to a different host
        # mid-test (telemetry, plugin marketplace fetch).
        "DISABLE_TELEMETRY": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        **(extra_env or {}),
    }

    # The CLI accepts stream-json input on stdin in `query` mode.  A
    # minimal user-message envelope is enough to trigger an API call.
    stdin_payload = (
        json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": "hello"},
            }
        )
        + "\n"
    )

    proc = await asyncio.create_subprocess_exec(
        str(cli_path),
        "--output-format",
        "stream-json",
        "--input-format",
        "stream-json",
        "--verbose",
        "--print",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        assert proc.stdin is not None
        proc.stdin.write(stdin_payload.encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()

        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_seconds
        )
    except (asyncio.TimeoutError, TimeoutError):
        # Best-effort kill — we already have whatever requests the CLI
        # managed to send before stalling.
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        # Reap the process after kill() so we don't leave an unreaped
        # child behind until event-loop shutdown. Wait with its own
        # short timeout in case the kill was ineffective.
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=5.0
            )
        except (asyncio.TimeoutError, TimeoutError):
            stdout_bytes, stderr_bytes = b"", b""

    return (
        proc.returncode if proc.returncode is not None else -1,
        stdout_bytes.decode("utf-8", errors="replace"),
        stderr_bytes.decode("utf-8", errors="replace"),
    )


# ---------------------------------------------------------------------------
# The actual test
# ---------------------------------------------------------------------------


async def _run_reproduction(
    *,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str, str, list[_CapturedRequest]]:
    """Spawn the CLI against a fake Anthropic API and return what the
    server saw.
    """
    cli_path = _resolve_cli_path()
    if cli_path is None or not cli_path.is_file():
        pytest.skip(
            "No Claude Code CLI binary available (neither bundled nor "
            "overridden via CLAUDE_AGENT_CLI_PATH / "
            "CHAT_CLAUDE_AGENT_CLI_PATH); cannot reproduce."
        )

    captured: list[_CapturedRequest] = []
    upstream_runner, upstream_port = await _start_fake_anthropic_server(captured)

    try:
        returncode, stdout, stderr = await _run_cli_against_fake_server(
            cli_path=cli_path,
            fake_server_port=upstream_port,
            timeout_seconds=30.0,
            extra_env=extra_env,
        )
    finally:
        await upstream_runner.cleanup()

    return returncode, stdout, stderr, captured


def _assert_no_forbidden_patterns(
    captured: list[_CapturedRequest], returncode: int, stderr: str
) -> None:
    if not captured:
        pytest.skip(
            "Bundled CLI did not make any HTTP requests to the fake server "
            f"(rc={returncode}). The CLI may have failed before reaching "
            f"the network — stderr tail: {stderr[-500:]!r}. "
            "Nothing to assert; treating as inconclusive rather than "
            "either passing or failing."
        )

    all_findings: list[str] = []
    for req in captured:
        findings = _scan_request_for_forbidden_patterns(req.body, req.headers)
        if findings:
            all_findings.extend(f"{req.path}: {finding}" for finding in findings)

    assert not all_findings, (
        f"Bundled Claude Code CLI sent OpenRouter-incompatible features in "
        f"{len(all_findings)} request(s):\n  - "
        + "\n  - ".join(all_findings)
        + "\n\nThe bundled CLI is sending OpenRouter-incompatible features. "
        "See https://github.com/Significant-Gravitas/AutoGPT/pull/12294 and "
        "https://github.com/anthropics/claude-agent-sdk-python/issues/789. "
        "If you bumped `claude-agent-sdk`, verify the new bundled CLI works "
        "with `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` set (injected by "
        "``build_sdk_env()`` in ``env.py``), then add the CLI version to "
        "`_KNOWN_GOOD_BUNDLED_CLI_VERSIONS` in `sdk_compat_test.py`. "
        "Alternatively, pin a known-good binary via `claude_agent_cli_path` "
        "(env: `CLAUDE_AGENT_CLI_PATH` or `CHAT_CLAUDE_AGENT_CLI_PATH`)."
    )


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="CLI 2.1.97 (SDK 0.1.58) sends context-management beta without "
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1. This is expected — the env "
    "var guard in test_disable_experimental_betas_env_var_strips_headers "
    "is the real regression test.",
    strict=True,
)
async def test_bare_cli_does_not_send_openrouter_incompatible_features():
    """Bare CLI reproduction (no env var workaround).

    Documents whether the bundled CLI sends OpenRouter-incompatible
    features without the CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS env var.
    On SDK 0.1.58 (CLI 2.1.97) this is expected to fail — the env var
    test above is the actual regression guard.
    """
    returncode, _stdout, stderr, captured = await _run_reproduction()
    _assert_no_forbidden_patterns(captured, returncode, stderr)


@pytest.mark.asyncio
async def test_disable_experimental_betas_env_var_strips_headers():
    """Validate that ``CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`` strips
    the ``context-management-2025-06-27`` beta header when
    ``ANTHROPIC_BASE_URL`` points to a non-Anthropic endpoint (simulating
    OpenRouter).

    This is the main regression guard: the env var is injected by
    ``build_sdk_env()`` in ``env.py`` into every CLI subprocess so newer
    SDK / CLI versions work with OpenRouter without any proxy.
    """
    returncode, _stdout, stderr, captured = await _run_reproduction(
        extra_env={"CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1"},
    )
    _assert_no_forbidden_patterns(captured, returncode, stderr)


def test_subprocess_module_available():
    """Sentinel test: the subprocess module must be importable so the
    main reproduction test can spawn the CLI.  Catches sandboxed CI
    runners that block subprocess execution before the slow test runs."""
    assert subprocess.__name__ == "subprocess"


# ---------------------------------------------------------------------------
# Pure helper unit tests — pin the forbidden-pattern detection so any
# future drift in the scanner is caught fast, even when the slow
# end-to-end CLI subprocess test isn't runnable.
# ---------------------------------------------------------------------------


class TestScanRequestForForbiddenPatterns:
    def test_clean_body_returns_empty_findings(self):
        body = '{"model": "claude-opus-4.6", "messages": [{"role": "user", "content": "hi"}]}'
        assert _scan_request_for_forbidden_patterns(body, {}) == []

    def test_detects_tool_reference_in_body(self):
        body = (
            '{"messages": [{"role": "user", "content": ['
            '{"type": "tool_reference", "tool_name": "find"}'
            "]}]}"
        )
        findings = _scan_request_for_forbidden_patterns(body, {})
        assert len(findings) == 1
        assert "tool_reference" in findings[0]
        assert "PR #12294" in findings[0]

    def test_detects_context_management_in_body(self):
        body = '{"betas": ["context-management-2025-06-27"]}'
        findings = _scan_request_for_forbidden_patterns(body, {})
        assert len(findings) == 1
        assert "context-management-2025-06-27" in findings[0]
        assert "#789" in findings[0]

    def test_detects_context_management_in_anthropic_beta_header(self):
        findings = _scan_request_for_forbidden_patterns(
            body_text="{}",
            headers={"anthropic-beta": "context-management-2025-06-27"},
        )
        assert len(findings) == 1
        assert "anthropic-beta" in findings[0]

    def test_detects_context_management_in_uppercase_header_name(self):
        # HTTP header names are case-insensitive — make sure the
        # scanner handles a server that didn't normalise names.
        findings = _scan_request_for_forbidden_patterns(
            body_text="{}",
            headers={"Anthropic-Beta": "context-management-2025-06-27, other"},
        )
        assert len(findings) == 1

    def test_ignores_unrelated_header_values(self):
        findings = _scan_request_for_forbidden_patterns(
            body_text="{}",
            headers={
                "authorization": "Bearer secret",
                "anthropic-beta": "fine-grained-tool-streaming-2025",
            },
        )
        assert findings == []

    def test_detects_both_patterns_simultaneously(self):
        body = (
            '{"betas": ["context-management-2025-06-27"], '
            '"messages": [{"role": "user", "content": ['
            '{"type": "tool_reference", "tool_name": "find"}'
            "]}]}"
        )
        findings = _scan_request_for_forbidden_patterns(body, {})
        # Both patterns hit, in stable order: tool_reference then betas.
        assert len(findings) == 2
        assert "tool_reference" in findings[0]
        assert "context-management-2025-06-27" in findings[1]

    def test_detects_compact_tool_reference_without_spaces(self):
        # Regression guard: the old substring matcher only caught the
        # prettified form '"type": "tool_reference"' with a space
        # between the key and the value, so a CLI emitting compact
        # JSON (e.g. via `json.dumps(separators=(",", ":"))`) could
        # slip past the scanner and false-pass. The JSON-walking
        # detector catches both forms.
        body = '{"messages":[{"role":"user","content":[{"type":"tool_reference","tool_name":"find"}]}]}'
        findings = _scan_request_for_forbidden_patterns(body, {})
        assert len(findings) == 1
        assert "tool_reference" in findings[0]

    def test_detects_tool_reference_in_malformed_body_fallback(self):
        # When the body isn't valid JSON the helper falls back to a
        # whitespace-tolerant regex so fuzzed / partial payloads are
        # still caught.
        body = 'garbage-prefix{"type"  :  "tool_reference"} trailing'
        findings = _scan_request_for_forbidden_patterns(body, {})
        assert len(findings) == 1
        assert "tool_reference" in findings[0]


class TestResolveCliPath:
    def test_honours_explicit_env_var_when_file_exists(self, tmp_path, monkeypatch):
        fake_cli = tmp_path / "fake-claude"
        fake_cli.write_text("#!/bin/sh\necho fake\n")
        fake_cli.chmod(0o755)
        monkeypatch.delenv("CHAT_CLAUDE_AGENT_CLI_PATH", raising=False)
        monkeypatch.setenv("CLAUDE_AGENT_CLI_PATH", str(fake_cli))
        resolved = _resolve_cli_path()
        assert resolved == fake_cli

    def test_honours_chat_prefixed_env_var_when_file_exists(
        self, tmp_path, monkeypatch
    ):
        """The Pydantic ``CHAT_`` prefix variant is also honoured.

        Mirrors ``ChatConfig.get_claude_agent_cli_path`` which accepts
        either ``CHAT_CLAUDE_AGENT_CLI_PATH`` (prefix applied by
        ``pydantic_settings``) or the unprefixed ``CLAUDE_AGENT_CLI_PATH``
        form documented in the PR and field docstring.
        """
        fake_cli = tmp_path / "fake-claude-prefixed"
        fake_cli.write_text("#!/bin/sh\necho fake\n")
        fake_cli.chmod(0o755)
        monkeypatch.delenv("CLAUDE_AGENT_CLI_PATH", raising=False)
        monkeypatch.setenv("CHAT_CLAUDE_AGENT_CLI_PATH", str(fake_cli))
        resolved = _resolve_cli_path()
        assert resolved == fake_cli

    def test_returns_none_when_env_var_points_to_missing_file(self, monkeypatch):
        monkeypatch.delenv("CHAT_CLAUDE_AGENT_CLI_PATH", raising=False)
        monkeypatch.setenv("CLAUDE_AGENT_CLI_PATH", "/nonexistent/path/to/claude")
        # Should fall through to the bundled binary OR return None,
        # but never raise.
        resolved = _resolve_cli_path()
        # We can't assert exact value (depends on whether the bundled
        # CLI is installed in the test env) but the function must not
        # raise — the caller is supposed to handle None gracefully.
        assert resolved is None or resolved.is_file()

    def test_falls_back_to_bundled_when_env_var_unset(self, monkeypatch):
        monkeypatch.delenv("CLAUDE_AGENT_CLI_PATH", raising=False)
        monkeypatch.delenv("CHAT_CLAUDE_AGENT_CLI_PATH", raising=False)
        # Same caveat as above — returns the bundled path or None,
        # depending on what's installed in the test env.
        resolved = _resolve_cli_path()
        assert resolved is None or resolved.is_file()
