"""A block given a model the catalog has switched off runs the model's catalog
fallback, whatever the value came from: the node's saved input, a workflow
input wired into its ``model`` pin, a preset, a schedule or an API call. A
model its provider still serves carries the provider's shutdown date, and runs
as itself until that day."""

import logging
from datetime import date, timedelta
from unittest.mock import patch

import pytest

from backend.blocks import _base
from backend.blocks.llm import AITextGeneratorBlock, LLMModel, LLMResponse
from backend.data.execution import ExecutionContext
from backend.data.llm_registry import llm_models
from backend.data.llm_registry.catalog import get_catalog
from backend.data.llm_registry.catalog_model import CatalogPayload
from backend.data.llm_registry.llm_models import (
    _retired_model_fallbacks,
    retired_model_fallback,
)
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
        async for _ in AITextGeneratorBlock().execute(
            {"prompt": "Say hello.", "model": model, "credentials": PLATFORM_OPENAI},
            credentials=openai_credentials,
            execution_context=ExecutionContext(user_id="user"),
        ):
            pass
    return asked


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
async def test_an_llm_block_given_an_unknown_model_still_fails():
    with pytest.raises(BlockInputError):
        await _run_ai_text_generator("gpt-0")


@pytest.mark.asyncio
async def test_a_swap_is_logged_once_per_process(on_day, monkeypatch, caplog):
    """A node left on a switched-off model logs its swap on the first run
    only, not on every run."""
    on_day(GPT_5_SHUTDOWN)
    monkeypatch.setattr(_base, "_LOGGED_MODEL_SWAPS", set())

    with caplog.at_level(logging.INFO, logger="backend.blocks._base"):
        assert await _run_ai_text_generator("o3-mini") == [LLMModel.GPT6_1_TERRA]
        assert await _run_ai_text_generator("o3-mini") == [LLMModel.GPT6_1_TERRA]

    swaps = [
        record.getMessage()
        for record in caplog.records
        if "in place of o3-mini" in record.getMessage()
    ]
    assert len(swaps) == 1
    assert "runs gpt-6.1-terra in place of o3-mini" in swaps[0]
