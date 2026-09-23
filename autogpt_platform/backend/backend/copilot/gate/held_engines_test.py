"""Each engine folds answered held calls into the turn it is starting.

One test per engine, driven through the engine's real entry point and stopped
at the fold: removing the fold from one engine turns only that engine's test
red. The held call itself runs inside ``resolve_answered``, so each test also
proves the turn's execution context is set before it runs.
"""

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.baseline.service import stream_chat_completion_baseline
from backend.copilot.context import get_execution_context, set_execution_context
from backend.copilot.model import ChatSession
from backend.copilot.model_router import ResolvedModel
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.sdk.expert_tool_gate_test import _make_patches, _make_session
from backend.copilot.sdk.tool_adapter import cap_late_tool_result

_RESULT = PendingMessage(
    content='<held_call_result tool="post_to_chat_platform">posted</held_call_result>'
)


class _StopAtFold(Exception):
    pass


@pytest.mark.asyncio
async def test_the_sdk_engine_opens_its_turn_with_the_held_result():
    from backend.copilot.sdk.service import stream_chat_completion_sdk

    order: list[str] = []
    folded: list[list[PendingMessage]] = []

    caps: list[object] = []

    async def resolve(*_args, **kwargs):
        order.append("resolve")
        caps.append(kwargs.get("cap"))
        return [_RESULT]

    async def persist(_session, _builder, pending, **_kwargs):
        folded.append(list(pending))
        raise _StopAtFold

    patches, _, _ = _make_patches(hire_experts_enabled=False)
    session = _make_session()
    with contextlib.ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        context = stack.enter_context(
            patch(
                "backend.copilot.sdk.service.set_execution_context",
                side_effect=lambda *a, **k: order.append("context"),
            )
        )
        stack.enter_context(
            patch("backend.copilot.sdk.service.resolve_answered", new=resolve)
        )
        stack.enter_context(
            patch(
                "backend.copilot.sdk.service.persist_pending_as_user_rows",
                new=persist,
            )
        )
        with contextlib.suppress(_StopAtFold):
            async for _ in stream_chat_completion_sdk(
                session_id=session.session_id,
                message="hello",
                is_user_message=True,
                user_id="test-user",
                session=session,
            ):
                pass

    context.assert_called()
    assert order.index("context") < order.index("resolve")
    assert folded == [[_RESULT]]
    # A late result is cut by the same rule as a direct MCP tool result.
    assert caps == [cap_late_tool_result]


@pytest.mark.asyncio
async def test_the_baseline_engine_opens_its_turn_with_the_held_result():
    session = ChatSession.new("user-1", dry_run=False)
    session.title = "already titled"
    seen_context: list[tuple[str | None, ChatSession | None]] = []
    sent: list[list[dict]] = []

    async def resolve(*_args, **_kwargs):
        seen_context.append(get_execution_context())
        return [_RESULT]

    async def model_loop(*, messages, **_kwargs):
        sent.append(list(messages))
        raise _StopAtFold
        yield

    svc = "backend.copilot.baseline.service"
    with (
        patch(f"{svc}.build_expert_identity_suffix", new=AsyncMock(return_value="")),
        patch(f"{svc}.drain_pending_safe", new=AsyncMock(return_value=[])),
        patch(
            f"{svc}._resolve_baseline_model",
            new=AsyncMock(
                return_value=ResolvedModel(
                    model="anthropic/claude-sonnet-4-6", source="env"
                )
            ),
        ),
        patch(
            f"{svc}.normalize_model_for_transport",
            new=MagicMock(side_effect=lambda model, cfg=None: model),
        ),
        patch(
            "backend.copilot.tools.e2b_sandbox.get_or_create_sandbox",
            new=AsyncMock(return_value=None),
        ),
        patch(
            f"{svc}._build_system_prompt",
            new=AsyncMock(return_value=("system prompt", None)),
        ),
        patch(f"{svc}.is_enabled_for_user", new=AsyncMock(return_value=False)),
        patch(f"{svc}.is_feature_enabled", new=AsyncMock(return_value=False)),
        patch(
            f"{svc}.build_builder_system_prompt_suffix",
            new=AsyncMock(return_value=""),
        ),
        patch(f"{svc}.extract_context_messages", new=AsyncMock(return_value=[])),
        patch(f"{svc}._compress_session_messages", new=AsyncMock(return_value=[])),
        patch(f"{svc}.resolve_answered", new=resolve),
        patch(f"{svc}.persist_pending_as_user_rows", new=AsyncMock(return_value=True)),
        patch(f"{svc}.tool_call_loop", new=model_loop),
    ):
        try:
            async for _ in stream_chat_completion_baseline(
                session_id=session.session_id,
                message=None,
                is_user_message=False,
                user_id="user-1",
                session=session,
            ):
                pass
        except Exception:
            pass  # what the turn does after the model's first call is not under test
        finally:
            set_execution_context(None, None)

    assert seen_context == [("user-1", session)]
    assert len(sent) == 1
    assert sent[0][-1] == {"role": "user", "content": _RESULT.content}
