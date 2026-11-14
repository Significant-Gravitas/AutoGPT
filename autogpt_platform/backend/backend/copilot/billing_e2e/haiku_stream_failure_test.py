import asyncio
import json
import uuid
from typing import Any, Literal
from unittest.mock import AsyncMock

import pytest
from aiohttp import web

from backend.copilot.baseline import service as baseline
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import StreamTextDelta, StreamToolOutputAvailable
from backend.copilot.tree import root_envelope
from backend.util.prompt import estimate_token_count, estimate_token_count_str

from . import provider as provider_module
from .baseline_turns import BaselineTurns
from .invoice import Route
from .ledger import PlatformLedger
from .provider import FakeProvider
from .wire import TOOL_NAME

MODEL = "claude-haiku-5-5"


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize(
    "requests,usage_mode,failure",
    [
        (requests, usage_mode, failure)
        for requests, usage_mode in [
            (1, "none"),
            (1, "all"),
            (2, "none"),
            (2, "all"),
            (2, "failed"),
            (2, "completed"),
        ]
        for failure in ["provider", "disconnect"]
    ]
    + [
        (1, "none", "provider_compat"),
        (2, "none", "provider_compat"),
        (2, "all", "provider_compat"),
        (3, "none", "provider"),
        (3, "all", "provider"),
        (1, "none", "user_stop"),
        (2, "none", "user_stop"),
    ],
)
async def test_provider_failures_are_excluded_but_disconnects_remain_billable(
    monkeypatch: pytest.MonkeyPatch,
    ledger: PlatformLedger,
    requests: int,
    failure: Literal["provider", "provider_compat", "disconnect", "user_stop"],
    usage_mode: Literal["none", "all", "failed", "completed"],
):
    config = baseline.config.model_copy(
        update={
            "use_local": False,
            "use_openrouter": False,
            "use_claude_code_subscription": False,
            "use_e2b_sandbox": False,
            "baseline_prompt_cache_ttl": "1h",
        }
    )
    async with FailingStreamProvider(requests, failure, usage_mode) as provider:
        run_turn = BaselineTurns(
            monkeypatch,
            config,
            base_url=f"{provider.url}/anthropic/v1",
            model=MODEL,
        )

        async def answer_tool(*, tool_call_id: str, **_: Any):
            return StreamToolOutputAvailable(
                toolCallId=tool_call_id,
                toolName=TOOL_NAME,
                output=" ".join(f"entry{i}" for i in range(12_000)),
            )

        monkeypatch.setattr(
            baseline, "execute_tool", AsyncMock(side_effect=answer_tool)
        )
        session = ChatSession.new("haiku-stream-failure", dry_run=False)
        session.title = "Stream failure regression"
        prompt = "Hello, please answer in detail"
        session.messages.append(ChatMessage(role="user", content=prompt))
        try:
            if failure == "user_stop":
                stream = baseline.stream_chat_completion_baseline(
                    session.session_id,
                    message=prompt,
                    is_user_message=True,
                    user_id=session.user_id,
                    session=session,
                    envelope=root_envelope(
                        str(uuid.uuid4()), session_id=session.session_id
                    ),
                )
                text_events = 0
                try:
                    async for event in stream:
                        if isinstance(event, StreamTextDelta):
                            text_events += 1
                            if text_events == requests:
                                break
                finally:
                    await stream.aclose()
                assert text_events == requests
            else:
                await run_turn(session, prompt, new_pod=False)
            await ledger.settle()
        finally:
            provider.release.set()
            await run_turn.aclose()

        assert len(provider.bodies) == requests
        prompt_estimates = [
            estimate_token_count(body["messages"], model=MODEL)
            + estimate_token_count_str(body["tools"], model=MODEL)
            for body in provider.bodies
        ]
        assert prompt_estimates[0] < 100_000
        if requests >= 2:
            assert prompt_estimates[1] > 100_000
        eligible = requests - int(failure in {"provider", "provider_compat"})
        reported = sum(provider.reported[:eligible])
        if reported:
            expected = reported * 4_750
        elif eligible:
            prompt_tokens = prompt_estimates[eligible - 1]
            output_tokens = estimate_token_count_str(
                "".join(provider.outputs[:eligible]), model=MODEL
            )
            prompt_rate = 0.5 if prompt_tokens > 100_000 else 0.1
            output_rate = 2.5 if prompt_tokens > 100_000 and eligible == 1 else 0.5
            expected = round(prompt_tokens * prompt_rate + output_tokens * output_rate)
        else:
            expected = 0
        tolerance = 1 if eligible else 0
        assert (
            abs(sum(charge.microdollars for charge in ledger.spend) - expected)
            <= tolerance
        )
        assert (
            abs(sum(charge.microdollars for charge in ledger.tree) - expected)
            <= tolerance
        )
        assert (
            abs(sum(row.entry.cost_microdollars for row in ledger.cost_log) - expected)
            <= tolerance
        )
        assert run_turn.client.is_closed()
        assert not provider.unexpected


class FailingStreamProvider(FakeProvider):
    def __init__(
        self,
        requests: int,
        failure: Literal["provider", "provider_compat", "disconnect", "user_stop"],
        usage_mode: Literal["none", "all", "failed", "completed"],
    ):
        super().__init__()
        self.requests = requests
        self.failure = failure
        self.usage_mode = usage_mode
        self.bodies: list[dict[str, Any]] = []
        self.outputs: list[str] = []
        self.reported: list[bool] = []
        self.release = asyncio.Event()

    async def _chat(self, request: web.Request, route: Route) -> web.StreamResponse:
        self.bodies.append(await request.json())
        assert len(self.bodies) <= self.requests
        last = len(self.bodies) == self.requests
        output = " amber" * 200
        self.outputs.append(output)
        base = {
            "id": f"chatcmpl-failure-{len(self.bodies)}",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": MODEL,
        }
        delta: dict[str, Any] = {"role": "assistant", "content": output}
        if not last:
            delta["tool_calls"] = [
                {
                    "index": i,
                    "id": f"call_stream_failure_{i}",
                    "type": "function",
                    "function": {
                        "name": TOOL_NAME,
                        "arguments": json.dumps({"query": f"item-{i}"}),
                    },
                }
                for i in range(3)
            ]
        chunks = [
            {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        ]
        report_usage = (
            self.usage_mode == "all"
            or (self.usage_mode == "failed" and last)
            or (self.usage_mode == "completed" and not last)
        )
        self.reported.append(report_usage)
        if report_usage:
            chunks.append(
                {
                    **base,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 90_000,
                        "completion_tokens": 300,
                        "total_tokens": 90_300,
                        "prompt_tokens_details": {
                            "cached_tokens": 60_000,
                            "cache_creation_input_tokens": 10_000,
                        },
                    },
                }
            )
        sse = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        if not last:
            finish = {
                **base,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            }
            return await provider_module._stream(
                request, sse + f"data: {json.dumps(finish)}\n\ndata: [DONE]\n\n"
            )
        if self.failure in {"provider", "provider_compat"}:
            error = {
                "type": "error",
                "error": {
                    "type": "overloaded_error",
                    "message": "Planned provider failure",
                },
            }
            if self.failure == "provider_compat":
                return await provider_module._stream(
                    request, sse + f"data: {json.dumps({'error': error['error']})}\n\n"
                )
            return await provider_module._stream(
                request, sse + f"event: error\ndata: {json.dumps(error)}\n\n"
            )
        data = sse.encode()
        headers = {"Content-Type": "text/event-stream"}
        if self.failure == "disconnect":
            headers["Content-Length"] = str(len(data) + 100)
        response = web.StreamResponse(headers=headers)
        await response.prepare(request)
        await response.write(data)
        if self.failure == "user_stop":
            await self.release.wait()
            return response
        await asyncio.sleep(0)
        assert request.transport is not None
        request.transport.close()
        return response
