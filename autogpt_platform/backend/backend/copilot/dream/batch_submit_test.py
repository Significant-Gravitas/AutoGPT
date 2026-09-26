"""Tests for dream batch submission.

Covers the orphan-prevention guard (a paid provider batch must be
cancelled when the BatchExecutor enqueue fails afterwards — otherwise it
runs to completion with no callback to consume it), the dream-lock
ownership token riding on the persisted input bundle so the batch
callback can compare-and-delete the lock hours later, and what a phase
submits: the native model spelling and a tool carrying its schema, forced
where the model accepts a forced tool.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.config import ChatConfig
from backend.copilot.dream.batch_submit import (
    INPUT_TTL_SECONDS,
    PHASE_DESCRIPTIONS,
    input_bundle_key,
    persist_input_bundle,
    phase_models_for_config,
    read_input_bundle,
    read_lock_token,
    submit_phase,
)
from backend.copilot.dream.fetch import DreamInput
from backend.copilot.dream.schemas import ConsolidationOutput
from backend.copilot.dream.structured_output import OUTPUT_TOOL_CALL_ONCE
from backend.copilot.graphiti.scope import DREAM_LOCK_KEY_PREFIX, MemoryScope
from backend.util.llm.providers import BatchSubmissionRef
from backend.util.llm.tool_use import (
    auto_tool_choice,
    force_tool_choice,
    pydantic_to_anthropic_tool,
)


def _bundle(expert_id: str | None = None) -> DreamInput:
    now = datetime.now(timezone.utc)
    return DreamInput(
        user_id="u1",
        expert_id=expert_id,
        group_id="user_u1" if expert_id is None else "expert_resolved",
        window_start=now,
        window_end=now,
    )


@pytest.fixture
def fake_redis():
    """Dict-backed redis stub matching the batch_callbacks_test pattern."""
    string_store: dict[str, str] = {}

    async def fake_get(key):
        return string_store.get(key)

    async def fake_set(key, value, ex=None, nx=False):
        if nx and key in string_store:
            return None
        string_store[key] = value
        return True

    async def fake_delete(key):
        string_store.pop(key, None)

    stub = AsyncMock()
    stub.get.side_effect = fake_get
    stub.set.side_effect = fake_set
    stub.delete.side_effect = fake_delete

    with patch(
        "backend.data.redis_client.get_redis_async",
        AsyncMock(return_value=stub),
    ):
        yield stub, string_store


@pytest.mark.asyncio
async def test_enqueue_failure_cancels_orphaned_batch_and_reraises(fake_redis):
    submitted = BatchSubmissionRef(
        provider="anthropic",
        provider_batch_id="msgbatch_orphan",
        custom_id="p1_consolidate",
        submitted_at=datetime.now(timezone.utc),
    )
    call = AsyncMock(return_value=submitted)
    enqueue = AsyncMock(side_effect=RuntimeError("redis down"))
    cancel = AsyncMock(return_value=True)

    with patch("backend.copilot.dream.batch_submit.call_provider", call), patch(
        "backend.copilot.dream.batch_submit.enqueue_pending", enqueue
    ), patch("backend.copilot.dream.batch_submit.cancel_batch", cancel):
        with pytest.raises(RuntimeError, match="redis down"):
            await submit_phase(
                user_id="u1",
                pass_id="p1",
                job_id="j1",
                phase="consolidate",
                phase_models={"consolidate": "claude-sonnet-4-6"},
                api_key="sk-test",
                input_bundle=_bundle(),
            )

    # The just-submitted, paid batch must be cancelled, not left orphaned.
    cancel.assert_awaited_once()
    assert cancel.call_args.kwargs["provider_batch_id"] == "msgbatch_orphan"
    assert cancel.call_args.kwargs["api_key"] == "sk-test"


@pytest.mark.asyncio
async def test_successful_enqueue_does_not_cancel(fake_redis):
    submitted = BatchSubmissionRef(
        provider="anthropic",
        provider_batch_id="msgbatch_ok",
        custom_id="p1_consolidate",
        submitted_at=datetime.now(timezone.utc),
    )
    call = AsyncMock(return_value=submitted)
    enqueue = AsyncMock(return_value=None)
    cancel = AsyncMock()

    with patch("backend.copilot.dream.batch_submit.call_provider", call), patch(
        "backend.copilot.dream.batch_submit.enqueue_pending", enqueue
    ), patch("backend.copilot.dream.batch_submit.cancel_batch", cancel):
        ref = await submit_phase(
            user_id="u1",
            pass_id="p1",
            job_id="j1",
            phase="consolidate",
            phase_models={"consolidate": "claude-sonnet-4-6"},
            api_key="sk-test",
            input_bundle=_bundle(),
        )

    enqueue.assert_awaited_once()
    cancel.assert_not_awaited()
    assert ref.provider_batch_id == "msgbatch_ok"
    assert enqueue.call_args.args[0].payload["expert_id"] is None


@pytest.mark.asyncio
async def test_submit_phase_refreshes_input_bundle_ttl(fake_redis):
    """Each phase batch gets its own 24h SLA window, so the input bundle's
    TTL — stamped once at persist — must be re-armed on every phase submit
    or a multi-phase chain can outlive its bundle and hard-fail the paid
    dream at the callback's read_input_bundle guard."""
    stub, _ = fake_redis
    submitted = BatchSubmissionRef(
        provider="anthropic",
        provider_batch_id="msgbatch_chain",
        custom_id="p1_recombine",
        submitted_at=datetime.now(timezone.utc),
    )
    with patch(
        "backend.copilot.dream.batch_submit.call_provider",
        AsyncMock(return_value=submitted),
    ), patch(
        "backend.copilot.dream.batch_submit.enqueue_pending",
        AsyncMock(return_value=None),
    ):
        await submit_phase(
            user_id="u1",
            pass_id="p1",
            job_id="j1",
            phase="recombine",
            phase_models={"recombine": "claude-opus-4-6"},
            api_key="sk-test",
            input_bundle=_bundle(),
            consolidated_json="{}",
        )

    stub.expire.assert_awaited_once_with(input_bundle_key("p1"), INPUT_TTL_SECONDS)


@pytest.mark.asyncio
async def test_expert_scope_round_trips_and_routes_live_lock_lookup(fake_redis):
    read_live_token = AsyncMock(return_value="tok-expert")
    with patch(
        "backend.copilot.dream.batch_submit.read_dream_lock_token",
        read_live_token,
    ):
        await persist_input_bundle("p-expert", _bundle("expert-1"))

    read_live_token.assert_awaited_once_with(MemoryScope.for_expert("u1", "expert-1"))
    bundle = await read_input_bundle("p-expert")
    assert bundle is not None
    assert bundle.expert_id == "expert-1"
    assert bundle.group_id == "expert_resolved"


@pytest.mark.asyncio
async def test_persist_input_bundle_carries_explicit_lock_token(fake_redis):
    """The orchestrator persists the bundle while still holding the dream
    lock and passes its OWN handle token; the bundle must carry it so the
    batch callback — hours later, in another process — can
    compare-and-delete."""
    _, string_store = fake_redis
    string_store[f"{DREAM_LOCK_KEY_PREFIX}u1"] = "tok-abc"

    await persist_input_bundle("p1", _bundle(), lock_token="tok-abc")

    assert await read_lock_token("p1") == "tok-abc"
    # The bundle itself still round-trips untouched by the extra field.
    bundle = await read_input_bundle("p1")
    assert bundle is not None
    assert bundle.user_id == "u1"


@pytest.mark.asyncio
async def test_persist_input_bundle_explicit_token_wins_over_live_key(fake_redis):
    """If this pass's lock expired and a NEWER pass re-acquired the key
    before persist runs, the live key holds the newer pass's token. The
    bundle must store the token the caller actually owns — storing the
    live value would let this pass's callback compare-and-delete the
    newer pass's lock hours later."""
    _, string_store = fake_redis
    string_store[f"{DREAM_LOCK_KEY_PREFIX}u1"] = "tok-newer-pass"

    await persist_input_bundle("p1", _bundle(), lock_token="tok-ours")

    assert await read_lock_token("p1") == "tok-ours"


@pytest.mark.asyncio
async def test_persist_input_bundle_falls_back_to_live_key_without_token(fake_redis):
    """Callers that don't supply a token (eval harness) fall back to
    reading the live lock key at persist time."""
    _, string_store = fake_redis
    string_store[f"{DREAM_LOCK_KEY_PREFIX}u1"] = "tok-live"

    await persist_input_bundle("p1", _bundle())

    assert await read_lock_token("p1") == "tok-live"


@pytest.mark.asyncio
async def test_persist_input_bundle_omits_token_when_lock_unheld(fake_redis):
    """No token supplied and no held lock at persist time ⇒ no token
    stored, and the callback falls back to the lock TTL instead of a
    blind delete."""
    await persist_input_bundle("p2", _bundle())

    assert await read_lock_token("p2") is None


@pytest.mark.asyncio
async def test_read_lock_token_none_when_bundle_expired(fake_redis):
    assert await read_lock_token("p-gone") is None


@pytest.mark.asyncio
async def test_read_lock_token_none_when_bundle_corrupted(fake_redis):
    _, string_store = fake_redis
    string_store[input_bundle_key("p3")] = "not json {{{"

    assert await read_lock_token("p3") is None


@pytest.mark.asyncio
async def test_submitted_phase_tool_carries_the_phase_schema(fake_redis):
    """The forced tool is built Anthropic-shaped (``input_schema``). The
    provider's tool conversion read only ``parameters``, so every phase
    went out with an empty schema and the model had to guess the shape."""
    create = AsyncMock(return_value=SimpleNamespace(id="msgbatch_schema"))
    client = SimpleNamespace(
        messages=SimpleNamespace(batches=SimpleNamespace(create=create))
    )
    with patch(
        "backend.util.llm.providers.anthropic.AsyncAnthropic", return_value=client
    ), patch(
        "backend.copilot.dream.batch_submit.enqueue_pending",
        AsyncMock(return_value=None),
    ):
        await submit_phase(
            user_id="u1",
            pass_id="p1",
            job_id="j1",
            phase="consolidate",
            phase_models={"consolidate": "claude-sonnet-5"},
            api_key="sk-test",
            input_bundle=_bundle(),
        )

    params = create.call_args.kwargs["requests"][0]["params"]
    expected = pydantic_to_anthropic_tool(
        ConsolidationOutput, tool_name="emit_consolidation", description=""
    )["input_schema"]
    (tool,) = params["tools"]
    assert tool["name"] == "emit_consolidation"
    assert tool["input_schema"]["properties"] == expected["properties"]
    assert tool["input_schema"]["required"] == expected.get("required", [])
    assert tool["input_schema"]["properties"]["facts"]["items"]["properties"]
    # A forced tool keeps its description as written.
    assert tool["description"] == PHASE_DESCRIPTIONS["consolidate"]
    assert params["tool_choice"] == force_tool_choice("emit_consolidation")
    assert params["model"] == "claude-sonnet-5"
    # A forced tool needs no prompt line asking for it.
    assert "emit_consolidation" not in params["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_opus_5_5_phase_offers_the_tool_under_auto(fake_redis):
    """Opus 5.5 answers a forced ``tool_choice`` with a 400, on the Batches
    API only hours later in the result row, so its phase goes out under
    ``auto``, the last user turn asking for one call with the full result."""
    create = AsyncMock(return_value=SimpleNamespace(id="msgbatch_auto"))
    client = SimpleNamespace(
        messages=SimpleNamespace(batches=SimpleNamespace(create=create))
    )
    with patch(
        "backend.util.llm.providers.anthropic.AsyncAnthropic", return_value=client
    ), patch(
        "backend.copilot.dream.batch_submit.enqueue_pending",
        AsyncMock(return_value=None),
    ):
        await submit_phase(
            user_id="u1",
            pass_id="p1",
            job_id="j1",
            phase="recombine",
            phase_models={"recombine": "claude-opus-5-5"},
            api_key="sk-test",
            input_bundle=_bundle(),
            consolidated_json='{"facts": []}',
        )

    params = create.call_args.kwargs["requests"][0]["params"]
    assert params["model"] == "claude-opus-5-5"
    assert params["tool_choice"] == auto_tool_choice()
    (tool,) = params["tools"]
    assert tool["name"] == "emit_recombination"
    assert tool["description"] == (
        f"{PHASE_DESCRIPTIONS['recombine']} {OUTPUT_TOOL_CALL_ONCE}"
    )
    last = params["messages"][-1]
    assert last["role"] == "user"
    assert "emit_recombination" in last["content"]
    # The ``$ref``'d enum field keeps its own default and description.
    proposal = tool["input_schema"]["properties"]["proposals"]["items"]
    memory_kind = proposal["properties"]["memory_kind"]
    assert "finding" in memory_kind["enum"]
    assert memory_kind["default"] == "finding"
    assert memory_kind["description"].startswith("Envelope kind")


def test_phase_models_take_the_native_anthropic_spelling():
    """The batch path submits to Anthropic directly even when chat runs on
    OpenRouter, so the OpenRouter spellings in config are converted."""
    config = ChatConfig.model_construct(
        fast_standard_model="anthropic/claude-sonnet-5",
        fast_advanced_model="anthropic/claude-opus-5.5",
    )
    assert phase_models_for_config(config) == {
        "consolidate": "claude-sonnet-5",
        "recombine": "claude-opus-5-5",
        "sanitize": "claude-sonnet-5",
    }


@pytest.mark.parametrize(
    "standard,advanced,named",
    [
        (
            "openai/gpt-4.1-mini",
            "anthropic/claude-opus-5.5",
            "CHAT_FAST_STANDARD_MODEL='openai/gpt-4.1-mini'",
        ),
        (
            "anthropic/claude-sonnet-5",
            "llama3.1:8b",
            "CHAT_FAST_ADVANCED_MODEL='llama3.1:8b'",
        ),
    ],
)
def test_phase_models_refuse_a_non_anthropic_model(
    standard: str, advanced: str, named: str
):
    """Refused at submit rather than failing in the batch results hours
    later, with the remedies this path has: it always submits to Anthropic
    directly, so enabling OpenRouter is not one of them."""
    config = ChatConfig.model_construct(
        fast_standard_model=standard, fast_advanced_model=advanced
    )
    with pytest.raises(ValueError) as exc_info:
        phase_models_for_config(config)
    message = str(exc_info.value)
    assert named in message
    assert "Choose Anthropic phase models" in message
    assert "disable batch routing" in message
    assert "dream-pass-batch-enabled" in message
    assert "OpenRouter" not in message
