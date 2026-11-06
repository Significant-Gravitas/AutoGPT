import os
from pathlib import Path

import pytest

from backend.copilot.sdk.cli_openrouter_compat_test import _resolve_cli_path
from backend.copilot.sdk.model_gateway.testing.canned_upstream import CannedReply
from backend.copilot.sdk.model_gateway.testing.cli_session import CliSession
from backend.copilot.sdk.model_gateway.testing.failure_streams import (
    NativeRetryUpstream,
    native_events,
)
from backend.copilot.sdk.model_gateway.testing.regressions import gateway_for
from backend.copilot.sdk.model_gateway.usage import GatewayUsage


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch) -> Path:
    path = _resolve_cli_path()
    assert path is not None, "The bundled or configured Claude Code CLI is required"
    for key in [k for k in os.environ if k.startswith(("CLAUDE", "ANTHROPIC"))]:
        monkeypatch.delenv(key)
    return path


async def test_real_cli_truncated_native_stream_bills_only_successful_retry(
    cli, tmp_path
):
    async with NativeRetryUpstream() as upstream, gateway_for(
        "anthropic_passthrough", upstream
    ) as gateway:
        session = CliSession(
            cli_path=cli,
            gateway=gateway,
            root=tmp_path,
            window_tokens=272_000,
            output_tokens=8_000,
        )
        (turn,) = await session.run(["Answer this short question."])
    assert not turn.is_error and turn.result == "retry succeeded"
    assert len(upstream.calls) == gateway.requests == 2
    assert gateway.usage.input_tokens == 100
    assert gateway.failed_usage.input_tokens == 100


@pytest.mark.parametrize("ending", ["eof", "error", "reset"])
async def test_real_cli_late_native_failure_retries_without_billing_failed_attempt(
    cli, tmp_path, ending
):
    events = native_events("already visible partial")[:-1]
    if ending == "error":
        events.append(
            {"type": "error", "error": {"type": "api_error", "message": "fake failure"}}
        )
    partial = CannedReply.of_sse(events)
    if ending == "reset":
        partial = partial.model_copy(update={"chunks": [partial.body]})
    async with NativeRetryUpstream(partial) as upstream, gateway_for(
        "anthropic_passthrough", upstream
    ) as gateway:
        session = CliSession(
            cli_path=cli,
            gateway=gateway,
            root=tmp_path,
            window_tokens=272_000,
            output_tokens=8_000,
        )
        (turn,) = await session.run(["Answer this short question."])
    assert not turn.is_error and turn.result == "retry succeeded"
    assert len(upstream.calls) == gateway.requests == 2
    assert gateway.usage == GatewayUsage(input_tokens=100, output_tokens=9)
    assert gateway.failed_usage == GatewayUsage(input_tokens=100, output_tokens=9)


async def test_real_cli_complete_native_stream_succeeds_once(cli, tmp_path):
    async with NativeRetryUpstream(
        CannedReply.of_sse(native_events("complete answer"))
    ) as upstream, gateway_for("anthropic_passthrough", upstream) as gateway:
        session = CliSession(
            cli_path=cli,
            gateway=gateway,
            root=tmp_path,
            window_tokens=272_000,
            output_tokens=8_000,
        )
        (turn,) = await session.run(["Answer this short question."])
    assert not turn.is_error and turn.result == "complete answer"
    assert len(upstream.calls) == gateway.requests == 1
    assert gateway.usage == GatewayUsage(input_tokens=100, output_tokens=9)
    assert gateway.failed_usage == GatewayUsage()
