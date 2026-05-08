"""Unit tests for title-generation cost tracking helpers.

Covers the new code added in PR #12882:
    * ``_title_usage_from_response`` — shape-robust OR ``usage.cost`` extraction
    * ``_record_title_generation_cost`` — provider-label + zero-tokens gate
    * ``_update_title_async`` — independent title / cost persistence try blocks
    * ``_generate_session_title`` — tuple return + robustness against empty choices

Mocks ``persist_and_record_usage`` / ``update_session_title`` at the boundary
where the code under test imports them (``backend.copilot.service.*``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from backend.copilot.service import (
    _generate_session_title,
    _record_title_generation_cost,
    _title_usage_from_response,
    _update_title_async,
)


def _build_completion(
    *,
    content: str | None = "Hello Title",
    usage: CompletionUsage | None = None,
    choices: list[Choice] | None = None,
) -> ChatCompletion:
    if choices is None:
        msg = ChatCompletionMessage(role="assistant", content=content)
        choices = [Choice(index=0, message=msg, finish_reason="stop")]
    return ChatCompletion(
        id="cmpl-1",
        choices=choices,
        created=0,
        model="anthropic/claude-haiku",
        object="chat.completion",
        usage=usage,
    )


def _usage_with_cost(cost: object | None) -> CompletionUsage:
    """Return a CompletionUsage whose ``model_extra`` carries ``cost``.

    Uses ``model_validate`` so OpenRouter's ``cost`` extension lands in
    the pydantic ``model_extra`` dict the helper reads from.
    """
    payload: dict[str, object] = {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
    }
    if cost is not None:
        payload["cost"] = cost
    return CompletionUsage.model_validate(payload)


class TestTitleUsageFromResponse:
    """``_title_usage_from_response`` returns sensible zeros/Nones when
    optional fields are absent or of unexpected shape."""

    def test_usage_none_returns_all_zero(self):
        resp = _build_completion(usage=None)
        prompt, completion, cache_read, cache_write, cost = _title_usage_from_response(
            resp
        )
        assert prompt == 0
        assert completion == 0
        assert cache_read == 0
        assert cache_write == 0
        assert cost is None

    def test_missing_cost_field_returns_none_cost(self):
        resp = _build_completion(usage=_usage_with_cost(None))
        prompt, completion, _, _, cost = _title_usage_from_response(resp)
        assert prompt == 12
        assert completion == 3
        assert cost is None

    def test_cost_as_int_is_coerced_to_float(self):
        resp = _build_completion(usage=_usage_with_cost(2))
        _, _, _, _, cost = _title_usage_from_response(resp)
        assert isinstance(cost, float)
        assert cost == 2.0

    def test_cost_as_float_is_returned_as_is(self):
        resp = _build_completion(usage=_usage_with_cost(0.000123))
        _, _, _, _, cost = _title_usage_from_response(resp)
        assert cost == pytest.approx(0.000123)

    def test_cost_as_non_numeric_string_returns_none(self):
        resp = _build_completion(usage=_usage_with_cost("free"))
        _, _, _, _, cost = _title_usage_from_response(resp)
        assert cost is None

    def test_empty_model_extra_returns_none_cost(self):
        # ``model_extra`` is empty for non-OR routes where pydantic didn't
        # receive any extras — prompt/completion still flow through.
        usage = CompletionUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7)
        resp = _build_completion(usage=usage)
        prompt, completion, _, _, cost = _title_usage_from_response(resp)
        assert (prompt, completion, cost) == (5, 2, None)

    def test_zero_prompt_and_completion_tokens(self):
        usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        resp = _build_completion(usage=usage)
        prompt, completion, _, _, cost = _title_usage_from_response(resp)
        assert (prompt, completion, cost) == (0, 0, None)

    def test_cached_tokens_extracted_from_prompt_tokens_details(self):
        from openai.types.completion_usage import PromptTokensDetails

        usage = CompletionUsage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=80),
        )
        resp = _build_completion(usage=usage)
        prompt, _, cache_read, _, _ = _title_usage_from_response(resp)
        assert prompt == 100
        assert cache_read == 80

    def test_cache_creation_tokens_via_anthropic_extras(self):
        from openai.types.completion_usage import PromptTokensDetails

        # Anthropic-native extras key on the OpenAI-compat endpoint.
        ptd = PromptTokensDetails.model_validate(
            {"cached_tokens": 50, "cache_creation_input_tokens": 30}
        )
        usage = CompletionUsage(
            prompt_tokens=200,
            completion_tokens=10,
            total_tokens=210,
            prompt_tokens_details=ptd,
        )
        resp = _build_completion(usage=usage)
        _, _, cache_read, cache_write, _ = _title_usage_from_response(resp)
        assert cache_read == 50
        assert cache_write == 30


class TestRecordTitleGenerationCost:
    """``_record_title_generation_cost`` persists cost + picks the right
    provider label and skips the DB roundtrip when nothing's meaningful
    to record."""

    @pytest.mark.asyncio
    async def test_openrouter_base_url_uses_open_router_provider(self):
        resp = _build_completion(usage=_usage_with_cost(0.0002))
        persist = AsyncMock(return_value=0)
        with (
            patch(
                "backend.copilot.service.persist_and_record_usage",
                new=persist,
            ),
            patch(
                "backend.copilot.service.config",
                MagicMock(
                    aux_provider_label="open_router",
                    title_model="anthropic/claude-haiku",
                ),
            ),
        ):
            await _record_title_generation_cost(
                response=resp, user_id="u", session_id="s"
            )
        persist.assert_awaited_once()
        kwargs = persist.await_args.kwargs
        assert kwargs["provider"] == "open_router"
        assert kwargs["model"] == "anthropic/claude-haiku"
        assert kwargs["prompt_tokens"] == 12
        assert kwargs["completion_tokens"] == 3
        assert kwargs["cost_usd"] == pytest.approx(0.0002)
        assert kwargs["log_prefix"] == "[title]"
        assert kwargs["session"] is None

    @pytest.mark.asyncio
    async def test_non_openrouter_base_url_uses_openai_provider(self):
        resp = _build_completion(usage=_usage_with_cost(0.0002))
        persist = AsyncMock(return_value=0)
        with (
            patch(
                "backend.copilot.service.persist_and_record_usage",
                new=persist,
            ),
            patch(
                "backend.copilot.service.config",
                MagicMock(
                    aux_provider_label="openai",
                    title_model="gpt-4o-mini",
                ),
            ),
        ):
            await _record_title_generation_cost(
                response=resp, user_id="u", session_id="s"
            )
        persist.assert_awaited_once()
        assert persist.await_args.kwargs["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_empty_base_url_uses_openai_provider(self):
        resp = _build_completion(usage=_usage_with_cost(0.0001))
        persist = AsyncMock(return_value=0)
        with (
            patch(
                "backend.copilot.service.persist_and_record_usage",
                new=persist,
            ),
            patch(
                "backend.copilot.service.config",
                MagicMock(aux_provider_label="openai", title_model="gpt-4o-mini"),
            ),
        ):
            await _record_title_generation_cost(
                response=resp, user_id=None, session_id=None
            )
        persist.assert_awaited_once()
        assert persist.await_args.kwargs["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_zero_tokens_zero_cost_skips_persist(self):
        """No cost, no tokens — the early return avoids a worthless
        ``PlatformCostLog`` row."""
        usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        resp = _build_completion(usage=usage)
        persist = AsyncMock(return_value=0)
        with (
            patch(
                "backend.copilot.service.persist_and_record_usage",
                new=persist,
            ),
            patch(
                "backend.copilot.service.config",
                MagicMock(
                    base_url="https://openrouter.ai/api/v1",
                    title_model="x",
                ),
            ),
        ):
            await _record_title_generation_cost(
                response=resp, user_id="u", session_id="s"
            )
        persist.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_usage_none_skips_persist(self):
        """``usage`` absent on the response == provider didn't report —
        still short-circuits to avoid writing a zero-valued row."""
        resp = _build_completion(usage=None)
        persist = AsyncMock(return_value=0)
        with (
            patch(
                "backend.copilot.service.persist_and_record_usage",
                new=persist,
            ),
            patch(
                "backend.copilot.service.config",
                MagicMock(
                    base_url="https://openrouter.ai/api/v1",
                    title_model="x",
                ),
            ),
        ):
            await _record_title_generation_cost(
                response=resp, user_id="u", session_id="s"
            )
        persist.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tokens_without_cost_still_records(self):
        """Tokens present but ``cost`` missing (non-OR route) still
        records a row so token counts are captured — ``cost_usd=None``
        is accepted by ``persist_and_record_usage``."""
        usage = CompletionUsage(prompt_tokens=8, completion_tokens=2, total_tokens=10)
        resp = _build_completion(usage=usage)
        persist = AsyncMock(return_value=0)
        with (
            patch(
                "backend.copilot.service.persist_and_record_usage",
                new=persist,
            ),
            patch(
                "backend.copilot.service.config",
                MagicMock(base_url=None, title_model="m"),
            ),
        ):
            await _record_title_generation_cost(
                response=resp, user_id="u", session_id="s"
            )
        persist.assert_awaited_once()
        assert persist.await_args.kwargs["cost_usd"] is None
        assert persist.await_args.kwargs["prompt_tokens"] == 8


class TestUpdateTitleAsync:
    """``_update_title_async`` runs title persistence and cost recording
    as independent best-effort steps — a failure in one does NOT
    cancel the other."""

    @pytest.mark.asyncio
    async def test_title_success_cost_success(self):
        resp = _build_completion(usage=_usage_with_cost(0.0001))
        gen = AsyncMock(return_value=("My Title", resp))
        update = AsyncMock(return_value=True)
        record = AsyncMock()
        with (
            patch(
                "backend.copilot.service._generate_session_title",
                new=gen,
            ),
            patch(
                "backend.copilot.service.update_session_title",
                new=update,
            ),
            patch(
                "backend.copilot.service._record_title_generation_cost",
                new=record,
            ),
        ):
            await _update_title_async("sess-1", "hello", user_id="u1")

        update.assert_awaited_once_with("sess-1", "u1", "My Title", only_if_empty=True)
        record.assert_awaited_once()
        assert record.await_args.kwargs["response"] is resp
        assert record.await_args.kwargs["user_id"] == "u1"
        assert record.await_args.kwargs["session_id"] == "sess-1"

    @pytest.mark.asyncio
    async def test_title_persist_fails_but_cost_still_recorded(self):
        resp = _build_completion(usage=_usage_with_cost(0.0001))
        gen = AsyncMock(return_value=("Title", resp))
        update = AsyncMock(side_effect=RuntimeError("prisma boom"))
        record = AsyncMock()
        with (
            patch(
                "backend.copilot.service._generate_session_title",
                new=gen,
            ),
            patch(
                "backend.copilot.service.update_session_title",
                new=update,
            ),
            patch(
                "backend.copilot.service._record_title_generation_cost",
                new=record,
            ),
        ):
            # Must NOT raise — persist failure is swallowed.
            await _update_title_async("sess-2", "msg", user_id="u")

        update.assert_awaited_once()
        record.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cost_record_fails_but_title_was_persisted(self):
        resp = _build_completion(usage=_usage_with_cost(0.0001))
        gen = AsyncMock(return_value=("Title", resp))
        update = AsyncMock(return_value=True)
        record = AsyncMock(side_effect=RuntimeError("cost record boom"))
        with (
            patch(
                "backend.copilot.service._generate_session_title",
                new=gen,
            ),
            patch(
                "backend.copilot.service.update_session_title",
                new=update,
            ),
            patch(
                "backend.copilot.service._record_title_generation_cost",
                new=record,
            ),
        ):
            # Must NOT raise — cost-recording failure is swallowed.
            await _update_title_async("sess-3", "msg", user_id="u")

        update.assert_awaited_once()
        record.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_user_id_skips_title_persist_but_records_cost(self):
        """Anonymous sessions skip the user-scoped title write, but we
        still paid for the LLM call — cost recording runs regardless."""
        resp = _build_completion(usage=_usage_with_cost(0.0001))
        gen = AsyncMock(return_value=("Title", resp))
        update = AsyncMock()
        record = AsyncMock()
        with (
            patch(
                "backend.copilot.service._generate_session_title",
                new=gen,
            ),
            patch(
                "backend.copilot.service.update_session_title",
                new=update,
            ),
            patch(
                "backend.copilot.service._record_title_generation_cost",
                new=record,
            ),
        ):
            await _update_title_async("sess-4", "msg", user_id=None)

        update.assert_not_awaited()
        record.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generation_returns_none_response_skips_cost(self):
        """``_generate_session_title`` swallows exceptions and returns
        ``(None, None)`` — no response means no cost to record."""
        gen = AsyncMock(return_value=(None, None))
        update = AsyncMock()
        record = AsyncMock()
        with (
            patch(
                "backend.copilot.service._generate_session_title",
                new=gen,
            ),
            patch(
                "backend.copilot.service.update_session_title",
                new=update,
            ),
            patch(
                "backend.copilot.service._record_title_generation_cost",
                new=record,
            ),
        ):
            await _update_title_async("sess-5", "msg", user_id="u")

        update.assert_not_awaited()
        record.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_title_with_response_still_records_cost(self):
        """Title came back empty but we still paid for the LLM call —
        cost recording runs even though the title write is skipped."""
        resp = _build_completion(usage=_usage_with_cost(0.0001))
        gen = AsyncMock(return_value=(None, resp))
        update = AsyncMock()
        record = AsyncMock()
        with (
            patch(
                "backend.copilot.service._generate_session_title",
                new=gen,
            ),
            patch(
                "backend.copilot.service.update_session_title",
                new=update,
            ),
            patch(
                "backend.copilot.service._record_title_generation_cost",
                new=record,
            ),
        ):
            await _update_title_async("sess-6", "msg", user_id="u")

        update.assert_not_awaited()
        record.assert_awaited_once()


class TestGenerateSessionTitle:
    """``_generate_session_title`` returns ``(title, response)`` — the
    caller owns both the persist and the cost-record decisions."""

    @pytest.mark.asyncio
    async def test_valid_response_returns_cleaned_title_and_response(self):
        # Code strips whitespace, then strips ``"'`` — whitespace inside
        # the quotes survives on purpose (titles like ``My Agent`` read
        # better than ``MyAgent``).  Test keeps the outer quotes + inner
        # whitespace distinct so the ordering is pinned.
        resp = _build_completion(content='"Clean Me"  ')
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=resp)
        with patch(
            "backend.copilot.service._get_aux_client",
            return_value=client,
        ):
            title, response = await _generate_session_title(
                "first message", user_id="u", session_id="s"
            )
        assert title == "Clean Me"
        assert response is resp

    @pytest.mark.asyncio
    async def test_long_title_truncated_with_ellipsis(self):
        """Titles >50 chars get truncated to 47 + '...'."""
        long_title = "A" * 80
        resp = _build_completion(content=long_title)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=resp)
        with patch(
            "backend.copilot.service._get_aux_client",
            return_value=client,
        ):
            title, _ = await _generate_session_title("x", user_id=None)
        assert title is not None
        assert len(title) == 50
        assert title.endswith("...")

    @pytest.mark.asyncio
    async def test_empty_choices_returns_none_title_with_response(self):
        """No ``choices`` on the response (shouldn't happen per SDK
        typing) must not raise IndexError — response is preserved so the
        caller can still record the paid-for cost."""
        resp = _build_completion(choices=[])
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=resp)
        with patch(
            "backend.copilot.service._get_aux_client",
            return_value=client,
        ):
            title, response = await _generate_session_title("x")
        assert title is None
        assert response is resp

    @pytest.mark.asyncio
    async def test_missing_message_returns_none_title(self):
        """A choice whose ``.message`` is absent produces a None title
        but the response still lands on the caller."""
        fake_choice = SimpleNamespace(message=None)
        fake_response = SimpleNamespace(choices=[fake_choice])
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=fake_response)
        with patch(
            "backend.copilot.service._get_aux_client",
            return_value=client,
        ):
            title, response = await _generate_session_title("x")
        assert title is None
        assert response is fake_response

    @pytest.mark.asyncio
    async def test_llm_call_raises_returns_none_none(self):
        """Network / API errors on the create call are swallowed;
        ``(None, None)`` ensures the caller skips both title and cost
        without crashing the background task."""
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("connection reset")
        )
        with patch(
            "backend.copilot.service._get_aux_client",
            return_value=client,
        ):
            title, response = await _generate_session_title("x")
        assert title is None
        assert response is None

    @pytest.mark.asyncio
    async def test_create_receives_usage_include_extra_body(self):
        """PR adds ``usage: {'include': True}`` so OpenRouter embeds the
        real billed cost into the final usage chunk — only when the aux
        transport is OR (Anthropic-direct rejects unknown extras)."""
        resp = _build_completion(content="Title")
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=resp)
        with (
            patch(
                "backend.copilot.service._get_aux_client",
                return_value=client,
            ),
            patch(
                "backend.copilot.service.config",
                MagicMock(
                    aux_uses_openrouter=True,
                    title_model="anthropic/claude-haiku",
                ),
            ),
        ):
            await _generate_session_title(
                "hello world", user_id="user-abc", session_id="sess-abc"
            )
        client.chat.completions.create.assert_awaited_once()
        extra_body = client.chat.completions.create.await_args.kwargs["extra_body"]
        assert extra_body["usage"] == {"include": True}
        assert extra_body["user"] == "user-abc"
        assert extra_body["session_id"] == "sess-abc"

    @pytest.mark.asyncio
    async def test_title_model_normalized_when_aux_is_anthropic(self):
        """When the aux client is pointed at api.anthropic.com (single-key
        direct-Anthropic deployment), the title model must have its
        ``anthropic/`` prefix and dot-separated version stripped before
        being sent — Anthropic's OpenAI-compat endpoint rejects both."""
        resp = _build_completion(content="Title")
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=resp)
        with (
            patch(
                "backend.copilot.service._get_aux_client",
                return_value=client,
            ),
            patch(
                "backend.copilot.service.config",
                MagicMock(
                    aux_uses_openrouter=False,
                    aux_provider_label="anthropic",
                    title_model="anthropic/claude-haiku-4.5",
                ),
            ),
        ):
            await _generate_session_title("hello", user_id=None, session_id=None)
        client.chat.completions.create.assert_awaited_once()
        kwargs = client.chat.completions.create.await_args.kwargs
        assert kwargs["model"] == "claude-haiku-4-5"

    @pytest.mark.asyncio
    async def test_title_model_unchanged_when_aux_is_openrouter(self):
        """OpenRouter routes by full ``vendor/model`` slug — the
        normalization branch must NOT fire for OR-routed aux."""
        resp = _build_completion(content="Title")
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=resp)
        with (
            patch(
                "backend.copilot.service._get_aux_client",
                return_value=client,
            ),
            patch(
                "backend.copilot.service.config",
                MagicMock(
                    aux_uses_openrouter=True,
                    aux_provider_label="open_router",
                    title_model="anthropic/claude-haiku-4.5",
                ),
            ),
        ):
            await _generate_session_title("hello", user_id=None, session_id=None)
        client.chat.completions.create.assert_awaited_once()
        kwargs = client.chat.completions.create.await_args.kwargs
        assert kwargs["model"] == "anthropic/claude-haiku-4.5"

    @pytest.mark.asyncio
    async def test_create_omits_extra_body_when_aux_not_openrouter(self):
        """When aux client is pointed at a non-OR endpoint (e.g.
        Anthropic OAI-compat), the OR-specific extras must not be sent
        — Anthropic's compat endpoint 400s on unknown fields."""
        resp = _build_completion(content="Title")
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=resp)
        with (
            patch(
                "backend.copilot.service._get_aux_client",
                return_value=client,
            ),
            patch(
                "backend.copilot.service.config",
                MagicMock(
                    aux_uses_openrouter=False,
                    title_model="anthropic/claude-haiku",
                ),
            ),
        ):
            await _generate_session_title(
                "hello world", user_id="user-abc", session_id="sess-abc"
            )
        client.chat.completions.create.assert_awaited_once()
        extra_body = client.chat.completions.create.await_args.kwargs["extra_body"]
        assert "usage" not in extra_body
        assert "user" not in extra_body
        assert "session_id" not in extra_body
