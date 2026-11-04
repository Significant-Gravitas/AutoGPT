import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiohttp import web

from backend.copilot.baseline import service as baseline
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import StreamToolOutputAvailable
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
    "answer_words,fail_followup", [(70_000, False), (12_000, True), (200, True)]
)
@pytest.mark.parametrize("successful_requests", [1, 2])
async def test_recovery_prices_the_sent_prompt_before_conversation_growth(
    monkeypatch: pytest.MonkeyPatch,
    ledger: PlatformLedger,
    answer_words: int,
    fail_followup: bool,
    successful_requests: int,
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
    async with GrowingConversationProvider(
        answer_words, fail_followup, successful_requests
    ) as provider:
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
                output=(
                    " ".join(f"entry{i}" for i in range(12_000))
                    if tool_call_id.startswith(f"call_recovery_{successful_requests}_")
                    else "The fact is 42."
                ),
            )

        monkeypatch.setattr(
            baseline, "execute_tool", AsyncMock(side_effect=answer_tool)
        )
        session = ChatSession.new("haiku-request-boundary", dry_run=False)
        session.title = "Request boundary regression"
        prompt = "Hello, please answer in detail"
        session.messages.append(ChatMessage(role="user", content=prompt))
        try:
            await run_turn(session, prompt, new_pod=False)
            await ledger.settle()
        finally:
            await run_turn.aclose()

        last_request = provider.bodies[successful_requests - 1]
        prompt_tokens = estimate_token_count(last_request["messages"], model=MODEL)
        tool_tokens = estimate_token_count_str(last_request["tools"], model=MODEL)
        output_tokens = estimate_token_count_str(provider.answer, model=MODEL)
        assert 0 < tool_tokens < 100_000 - prompt_tokens
        assert len(provider.bodies) == successful_requests + int(fail_followup)
        if fail_followup:
            assert (
                estimate_token_count(provider.bodies[-1]["messages"], model=MODEL)
                > 100_000
            )
        else:
            assert output_tokens > 100_000
        expected = round((prompt_tokens + tool_tokens) * 0.1 + output_tokens * 0.5)
        assert [charge.microdollars for charge in ledger.spend] == [expected]
        assert [charge.microdollars for charge in ledger.tree] == [expected]
        assert run_turn.client.is_closed()
        assert not provider.unexpected


class GrowingConversationProvider(FakeProvider):
    def __init__(
        self, answer_words: int, fail_followup: bool, successful_requests: int
    ):
        super().__init__()
        self.answer = " amber" * answer_words
        self.fail_followup = fail_followup
        self.successful_requests = successful_requests
        self.bodies: list[dict[str, Any]] = []

    async def _chat(self, request: web.Request, route: Route) -> web.StreamResponse:
        self.bodies.append(await request.json())
        if len(self.bodies) > self.successful_requests:
            return web.json_response(
                {"error": {"message": "Planned follow-up refusal"}}, status=400
            )
        base = {
            "id": "chatcmpl-recovery-boundary",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": MODEL,
        }
        final_request = len(self.bodies) == self.successful_requests
        calls_tools = not final_request or self.fail_followup
        delta: dict[str, Any] = {
            "role": "assistant",
            "content": self.answer if final_request else "",
        }
        if calls_tools:
            delta["tool_calls"] = [
                {
                    "index": i,
                    "id": f"call_recovery_{len(self.bodies)}_{i}",
                    "type": "function",
                    "function": {
                        "name": TOOL_NAME,
                        "arguments": json.dumps({"query": f"item-{i}"}),
                    },
                }
                for i in range(3 if final_request else 1)
            ]
        finish = "tool_calls" if calls_tools else "stop"
        chunks = [
            {
                **base,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
        ]
        sse = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return await provider_module._stream(request, sse + "data: [DONE]\n\n")
