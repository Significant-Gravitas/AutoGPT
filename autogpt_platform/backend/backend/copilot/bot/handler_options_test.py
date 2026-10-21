"""Handler routing of ``expert=`` / ``mode=`` and slash-command options."""

import dataclasses
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.platform_linking.models import BotExpert, EnsureSessionResult

from .adapters.base import InboundAttachment, MessageContext
from .bot_backend import ResolveResult
from .expert_choice import SERVER_EXPERTS_REPLY
from .handler import MessageHandler, TargetState, with_typed_options
from .options import TurnOptions

BOLT = BotExpert(id="e1", name="Bolt", role="Builder")
BOLTZ = BotExpert(id="e2", name="Boltzmann", role="Physicist")
SAGE = BotExpert(id="e3", name="Sage", role="Researcher")


def _ctx(text: str, *, dm: bool = True, **kw) -> MessageContext:
    return MessageContext(
        platform="discord",
        channel_type="dm" if dm else "channel",
        server_id=None if dm else "guild-1",
        channel_id="chan-1",
        message_id="msg-1",
        user_id="user-1",
        username="Bently",
        text=text,
        bot_mentioned=not dm,
        **kw,
    )


def _api(experts: list[BotExpert] | None = None) -> MagicMock:
    api = MagicMock()
    api.resolve_server = AsyncMock(return_value=ResolveResult(linked=True))
    api.resolve_user = AsyncMock(return_value=ResolveResult(linked=True))
    api.list_user_experts = AsyncMock(return_value=experts or [])
    api.upload_workspace_files = AsyncMock(return_value=[])
    api.ensure_session = AsyncMock(return_value=EnsureSessionResult(session_id="s1"))
    return api


def _adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.send_message = AsyncMock()
    adapter.send_reply = AsyncMock()
    adapter.max_attachment_bytes = 25 * 1024 * 1024
    return adapter


def _capture(handler: MessageHandler) -> list[dict]:
    calls: list[dict] = []

    async def fake_stream_batch(batch, ctx, ad, tid, **kwargs):
        calls.append({"batch": list(batch), "target": tid, **kwargs})

    handler._stream_batch = fake_stream_batch  # type: ignore[method-assign]
    return calls


class TestTypedOptions:
    @pytest.mark.asyncio
    async def test_dm_expert_and_mode_route_the_turn(self):
        api = _api([BOLT, SAGE])
        handler = MessageHandler(api)
        calls = _capture(handler)
        await handler.handle(_ctx("expert=bolt mode=smart fix it"), _adapter())

        assert len(calls) == 1
        assert calls[0]["batch"] == [("Bently", "user-1", "fix it")]
        assert calls[0]["options"] == TurnOptions(
            expert_name="bolt", expert_id="e1", model="advanced"
        )
        api.list_user_experts.assert_awaited_once_with("discord", "user-1", None)

    @pytest.mark.asyncio
    async def test_mode_alone_needs_no_expert_lookup(self):
        api = _api()
        handler = MessageHandler(api)
        calls = _capture(handler)
        await handler.handle(_ctx("mode=fast hi"), _adapter())

        assert calls[0]["options"] == TurnOptions(model="standard")
        api.list_user_experts.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_server_expert_request_is_refused_in_place(self):
        api = _api([BOLT])
        handler = MessageHandler(api)
        calls = _capture(handler)
        adapter = _adapter()
        await handler.handle(_ctx("expert=Bolt hi", dm=False), adapter)

        assert calls == []
        adapter.send_reply.assert_awaited_once_with(
            "chan-1", SERVER_EXPERTS_REPLY, "msg-1"
        )
        api.list_user_experts.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_expert_lists_the_available_ones(self):
        handler = MessageHandler(_api([BOLT, SAGE]))
        calls = _capture(handler)
        adapter = _adapter()
        await handler.handle(_ctx("expert=Nova hi"), adapter)

        assert calls == []
        reply = adapter.send_message.await_args.args[1]
        assert 'No expert called "Nova"' in reply
        assert "Bolt (Builder)" in reply and "Sage (Researcher)" in reply

    @pytest.mark.asyncio
    async def test_ambiguous_prefix_lists_the_matches(self):
        handler = MessageHandler(_api([BOLT, BOLTZ, SAGE]))
        calls = _capture(handler)
        adapter = _adapter()
        await handler.handle(_ctx("expert=bol hi"), adapter)

        assert calls == []
        reply = adapter.send_message.await_args.args[1]
        assert "More than one expert" in reply
        assert "Boltzmann" in reply and "Sage" not in reply

    @pytest.mark.asyncio
    async def test_exact_name_beats_prefix(self):
        handler = MessageHandler(_api([BOLT, BOLTZ]))
        calls = _capture(handler)
        await handler.handle(_ctx("expert=Bolt hi"), _adapter())
        assert calls[0]["options"].expert_id == "e1"

    @pytest.mark.asyncio
    async def test_no_hires_says_so(self):
        handler = MessageHandler(_api([]))
        _capture(handler)
        adapter = _adapter()
        await handler.handle(_ctx("expert=Bolt hi"), adapter)
        assert "haven't hired" in adapter.send_message.await_args.args[1]

    @pytest.mark.asyncio
    async def test_attachments_upload_into_the_expert_session(self):
        api = _api([BOLT])
        handler = MessageHandler(api)
        _capture(handler)
        attachment = InboundAttachment(
            filename="a.txt", content=b"x", mime_type="text/plain"
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "backend.copilot.bot.handler.sessions.get_session",
                AsyncMock(return_value=None),
            )
            mp.setattr("backend.copilot.bot.handler.sessions.set_session", AsyncMock())
            await handler.handle(
                _ctx("expert=Bolt read this", attachments=(attachment,)), _adapter()
            )
        assert api.ensure_session.await_args.kwargs["expert_id"] == "e1"


class TestCommandOptions:
    @pytest.mark.asyncio
    async def test_command_expert_id_skips_name_lookup(self):
        api = _api([BOLT])
        handler = MessageHandler(api)
        calls = _capture(handler)
        ctx = _ctx("hi", options=TurnOptions(expert_id="e1", expert_name="Bolt"))
        await handler.handle(ctx, _adapter())

        assert calls[0]["options"].expert_id == "e1"
        api.list_user_experts.assert_not_awaited()

    def test_typed_expert_overrides_the_command_pick(self):
        ctx = _ctx(
            "expert=Sage hi", options=TurnOptions(expert_id="e1", model="advanced")
        )
        out = with_typed_options(ctx)
        assert out.text == "hi"
        assert out.options == TurnOptions(expert_name="Sage", model="advanced")

    def test_plain_text_keeps_the_context(self):
        ctx = _ctx("hi there")
        assert with_typed_options(ctx) is ctx


class TestBatchedOptions:
    @pytest.mark.asyncio
    async def test_follow_ups_keep_the_batch_options(self):
        handler = MessageHandler(_api())
        calls = _capture(handler)
        state = TargetState(processing=True)
        handler._targets["t"] = state
        first = dataclasses.replace(
            _ctx("one"), options=TurnOptions(expert_id="e1", model="advanced")
        )
        await handler._enqueue_and_process(first, _adapter(), "t")
        await handler._enqueue_and_process(_ctx("two"), _adapter(), "t")
        assert calls == []

        state.processing = False
        await handler._process(_ctx("x"), _adapter(), "t", state)

        assert len(calls) == 1
        assert [m[2] for m in calls[0]["batch"]] == ["one", "two"]
        assert calls[0]["options"] == TurnOptions(expert_id="e1", model="advanced")
        assert "t" not in handler._targets
