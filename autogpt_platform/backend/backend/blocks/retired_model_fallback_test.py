"""A block given a model the catalog has switched off runs the model's catalog
fallback, whatever the value came from: the node's saved input, a workflow
input wired into its ``model`` pin, a preset, a schedule or an API call. A
model its provider still serves carries the provider's shutdown date, and runs
as itself until that day."""

import logging
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from backend.blocks import get_blocks
from backend.blocks.llm import AITextGeneratorBlock, LLMModel, LLMResponse
from backend.data.execution import ExecutionContext
from backend.data.graph import Link, Node
from backend.data.llm_registry import llm_models
from backend.data.llm_registry.catalog import get_catalog
from backend.data.llm_registry.catalog_model import CatalogPayload
from backend.data.llm_registry.llm_models import (
    _retired_model_fallbacks,
    is_llm_model_field,
    retired_model_fallback,
)
from backend.data.model import NodeExecutionStats
from backend.executor.manager import _validate_queued_input
from backend.executor.utils import block_usage_cost
from backend.integrations.credentials_store import openai_credentials
from backend.util.exceptions import BlockInputError

PLATFORM_OPENAI = {
    "id": openai_credentials.id,
    "provider": "openai",
    "type": "api_key",
    "title": openai_credentials.title,
}
# OpenAI shuts the GPT-5 and o3 models down on this day (their shutdown_at).
GPT_5_SHUTDOWN = date(2026, 12, 11)


@pytest.fixture
def on_day(monkeypatch: pytest.MonkeyPatch):
    """Freeze the day a shutdown date is read against."""

    def freeze(day: date) -> None:
        monkeypatch.setattr(llm_models, "_today", lambda: day)

    return freeze


@pytest.mark.parametrize(
    ("value", "runs"),
    [
        # Still served until its shutdown date, which has come.
        ("gpt-5-2025-08-07", LLMModel.GPT6_1_SOL),
        ("openai/gpt-5-2025-08-07", LLMModel.GPT6_1_SOL),
        # Already shut down by OpenAI: no date, the fallback at once.
        ("o3-mini", LLMModel.GPT6_1_TERRA),
        # Switched off before this change, with a fallback already set.
        ("mistralai/mistral-medium-3.1", LLMModel.MISTRAL_MEDIUM_3_5),
        # A model the catalog serves runs as it is.
        ("gpt-6.1-sol", None),
        # An unknown value isn't redirected; the block refuses it.
        ("gpt-0", None),
        (None, None),
    ],
)
def test_a_switched_off_model_resolves_to_its_catalog_fallback(
    value: str | None, runs: LLMModel | None, on_day
):
    on_day(GPT_5_SHUTDOWN)
    assert retired_model_fallback(value) == runs


@pytest.mark.parametrize(
    "slug",
    sorted(m.slug for m in get_catalog().models if m.shutdown_at is not None),
)
def test_a_still_served_model_runs_as_itself_until_its_shutdown_date(slug: str, on_day):
    model = next(m for m in get_catalog().models if m.slug == slug)
    assert model.shutdown_at is not None and model.fallback_model_slug
    fallback = LLMModel(model.fallback_model_slug)

    on_day(model.shutdown_at - timedelta(days=1))
    assert retired_model_fallback(slug) is None
    on_day(model.shutdown_at)
    assert retired_model_fallback(slug) == fallback
    on_day(model.shutdown_at + timedelta(days=1))
    assert retired_model_fallback(slug) == fallback


def test_a_model_already_shut_down_runs_its_fallback_on_any_day(on_day):
    on_day(date(2026, 1, 1))
    assert retired_model_fallback("o3-mini") == LLMModel.GPT6_1_TERRA


def _catalog_with_fallback(slug: str, fallback: str) -> CatalogPayload:
    catalog = get_catalog()
    return catalog.model_copy(
        update={
            "models": [
                (
                    model.model_copy(update={"fallback_model_slug": fallback})
                    if model.slug == slug
                    else model
                )
                for model in catalog.models
            ]
        }
    )


@pytest.mark.parametrize(
    ("slug", "fallback"),
    [
        # Another provider's model: the node's credentials don't fit it.
        ("gpt-5-2025-08-07", "claude-sonnet-5-5"),
        # A fallback the catalog has switched off too.
        ("gpt-5-2025-08-07", "o3-2025-04-16"),
        # A model the catalog serves runs as it is, fallback or not.
        ("gpt-6.1-sol", "gpt-6.1-terra"),
    ],
    ids=["other-provider", "switched-off-fallback", "served-model"],
)
def test_a_block_runs_a_fallback_only_for_a_switched_off_model_and_only_one_it_can_run(
    slug: str, fallback: str
):
    assert LLMModel(slug) not in _retired_model_fallbacks(
        _catalog_with_fallback(slug, fallback)
    )
    # The unchanged catalog: only the switched-off model has a fallback run.
    assert (LLMModel(slug) in _retired_model_fallbacks(get_catalog())) == (
        slug == "gpt-5-2025-08-07"
    )


async def _run_ai_text_generator(model: str) -> list[LLMModel]:
    return await _run_block(
        {"prompt": "Say hello.", "model": model, "credentials": PLATFORM_OPENAI}
    )


async def _run_block(
    input_data: dict, block: AITextGeneratorBlock | None = None
) -> list[LLMModel]:
    """Run the block for real down to the provider call, and return the model
    each call asked for."""
    asked: list[LLMModel] = []

    async def provider_call(*, llm_model: LLMModel, prompt: list, **kwargs):
        asked.append(llm_model)
        return LLMResponse(
            raw_response=None,
            prompt=prompt,
            response="Hello.",
            tool_calls=None,
            prompt_tokens=1,
            completion_tokens=1,
        )

    with patch("backend.blocks.llm.llm_call", provider_call):
        async for _ in (block or AITextGeneratorBlock()).execute(
            input_data,
            credentials=openai_credentials,
            execution_context=ExecutionContext(user_id="user"),
        ):
            pass
    return asked


_SCHEMA_DEFAULT = AITextGeneratorBlock.Input.model_fields["model"].default


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("saved", "day", "runs"),
    [
        ("gpt-5-2025-08-07", GPT_5_SHUTDOWN - timedelta(days=1), LLMModel.GPT5),
        ("openai/gpt-5-2025-08-07", GPT_5_SHUTDOWN, LLMModel.GPT6_1_SOL),
        (None, GPT_5_SHUTDOWN, _SCHEMA_DEFAULT),
    ],
    ids=["before-the-date", "on-the-day", "schema-default"],
)
async def test_a_run_records_the_model_it_runs_as_it_starts(
    saved: str | None, day: date, runs: LLMModel, on_day
):
    """The run's stats name the model it runs, in the enum's spelling, so its
    settlement and cost log price and name that model on any later day,
    and the run's record shows it. Its inputs keep the stored value."""
    on_day(day)
    block = AITextGeneratorBlock()
    input_data = {"prompt": "Say hello.", "credentials": PLATFORM_OPENAI}
    if saved is not None:
        input_data["model"] = saved

    assert await _run_block(input_data, block) == [runs]
    assert block.execution_stats.resolved_model == runs.value
    assert input_data.get("model") == saved


def test_a_block_has_at_most_one_llm_model_input():
    """A run records one model (``NodeExecutionStats.resolved_model``), and
    every ``LLMModel`` input of the block is priced as that model."""
    llm_model_inputs = {
        block.name: [
            name
            for name, field in block.input_schema.model_fields.items()
            if is_llm_model_field(field)
        ]
        for block in (block_cls() for block_cls in get_blocks().values())
    }
    assert {
        name: inputs for name, inputs in llm_model_inputs.items() if len(inputs) > 1
    } == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("saved", "day", "runs"),
    [
        # Still served the day before its shutdown date: runs as itself.
        ("gpt-5-2025-08-07", GPT_5_SHUTDOWN - timedelta(days=1), LLMModel.GPT5),
        # From that day on, on its fallback.
        ("gpt-5-2025-08-07", GPT_5_SHUTDOWN, LLMModel.GPT6_1_SOL),
        ("o3-mini", GPT_5_SHUTDOWN - timedelta(days=1), LLMModel.GPT6_1_TERRA),
        ("gpt-6.1-sol", GPT_5_SHUTDOWN, LLMModel.GPT6_1_SOL),
    ],
    ids=["gpt-5-before", "gpt-5-on-the-day", "o3-mini", "served"],
)
async def test_an_llm_block_given_a_switched_off_model_runs_its_fallback(
    saved: str, day: date, runs: LLMModel, on_day
):
    on_day(day)
    assert await _run_ai_text_generator(saved) == [runs]


@pytest.mark.asyncio
async def test_a_model_wired_in_from_a_workflow_input_runs_and_is_charged_as_its_fallback(
    on_day,
):
    """A workflow input wired into the ``model`` pin brings its value when the
    run starts, where no pass over stored graphs reaches it. The node is queued
    with the value as sent, then runs, and is priced as, the fallback."""
    on_day(GPT_5_SHUTDOWN)
    block = AITextGeneratorBlock()
    node = Node(
        id="llm-node",
        block_id=block.id,
        input_default={"prompt": "Say hello.", "credentials": PLATFORM_OPENAI},
        input_links=[
            Link(
                source_id="model-input",
                sink_id="llm-node",
                source_name="result",
                sink_name="model",
            )
        ],
    )

    queued, _, input_error = _validate_queued_input(
        node, {"model": "gpt-5-2025-08-07"}, dry_run=False
    )

    assert queued is not None and input_error is None
    assert queued["model"] == "gpt-5-2025-08-07"
    assert await _run_block(dict(queued)) == [LLMModel.GPT6_1_SOL]
    stats = NodeExecutionStats(input_token_count=1_000_000, output_token_count=0)
    _, matched = block_usage_cost(block, queued, stats=stats)
    assert matched["model"] == LLMModel.GPT6_1_SOL


@pytest.mark.asyncio
async def test_an_llm_block_given_an_unknown_model_still_fails():
    with pytest.raises(BlockInputError):
        await _run_ai_text_generator("gpt-0")


def test_the_day_a_shutdown_date_is_read_against_is_the_utc_day(monkeypatch):
    """23:30 UTC on 10 Dec is already 11 Dec on a host east of UTC+0:30; the
    gate must still read 10 Dec."""
    instant = datetime(2026, 12, 10, 23, 30, tzinfo=timezone.utc)

    class HostEastOfUtc(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:  # a naive now() is the host's local clock: UTC+14 here
                return (instant + timedelta(hours=14)).replace(tzinfo=None)
            return instant.astimezone(tz)

    monkeypatch.setattr(llm_models, "datetime", HostEastOfUtc)
    assert llm_models._today() == date(2026, 12, 10)
    assert retired_model_fallback("gpt-5-2025-08-07") is None


async def _run_and_log(model: str, caplog) -> list[str]:
    async def provider_call(*, llm_model: LLMModel, prompt: list, **kwargs):
        return LLMResponse(
            raw_response=None,
            prompt=prompt,
            response="Hello.",
            tool_calls=None,
            prompt_tokens=1,
            completion_tokens=1,
        )

    with (
        patch("backend.blocks.llm.llm_call", provider_call),
        caplog.at_level(logging.INFO, logger="backend.blocks._base"),
    ):
        async for _ in AITextGeneratorBlock().execute(
            {"prompt": "hi", "model": model, "credentials": PLATFORM_OPENAI},
            credentials=openai_credentials,
            execution_context=ExecutionContext(user_id="user", node_exec_id="ne-1"),
        ):
            pass
    return [
        record.getMessage()
        for record in caplog.records
        if "catalog fallbacks" in record.getMessage()
    ]


@pytest.mark.asyncio
async def test_a_swapped_run_logs_the_model_swap(on_day, caplog):
    """Each swapped run logs one INFO line with its node execution id: the
    run's recorded inputs keep the stored model, so this is its record."""
    on_day(GPT_5_SHUTDOWN)
    assert await _run_and_log("gpt-5-2025-08-07", caplog) == [
        "Block AITextGeneratorBlock runs switched-off models' catalog fallbacks "
        "(node exec ne-1): model gpt-5-2025-08-07 -> gpt-6.1-sol"
    ]


@pytest.mark.asyncio
async def test_the_swap_log_names_the_model_not_the_stored_spelling(on_day, caplog):
    on_day(GPT_5_SHUTDOWN)
    assert await _run_and_log("weird prefix/o3-mini", caplog) == [
        "Block AITextGeneratorBlock runs switched-off models' catalog fallbacks "
        "(node exec ne-1): model o3-mini -> gpt-6.1-terra"
    ]


@pytest.mark.asyncio
async def test_a_value_too_deep_to_resolve_still_fails_the_input_check(on_day):
    """A thousand provider prefixes recurse through ``LLMModel._missing_``. The
    value is unknown, and the block's input check refuses it, as before."""
    on_day(GPT_5_SHUTDOWN)
    deep = "x/" * 1000 + "o3-mini"
    assert retired_model_fallback(deep) is None
    with pytest.raises(BlockInputError):
        await _run_ai_text_generator(deep)
