"""A block given a model the catalog has switched off runs the model's catalog
fallback, whatever the value came from: the node's saved input, a workflow
input wired into its ``model`` pin, a preset, a schedule or an API call."""

from unittest.mock import patch

import pytest

from backend.blocks.llm import AITextGeneratorBlock, LLMModel, LLMResponse
from backend.data.execution import ExecutionContext
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


@pytest.mark.parametrize(
    ("value", "runs"),
    [
        # Still served until 2026-12-11, switched off now.
        ("gpt-5-2025-08-07", LLMModel.GPT6_1_SOL),
        ("openai/gpt-5-2025-08-07", LLMModel.GPT6_1_SOL),
        # Already shut down by OpenAI.
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
    value: str | None, runs: LLMModel | None
):
    assert retired_model_fallback(value) == runs


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
    ("saved", "runs"),
    [
        ("gpt-5-2025-08-07", LLMModel.GPT6_1_SOL),
        ("o3-mini", LLMModel.GPT6_1_TERRA),
        ("gpt-6.1-sol", LLMModel.GPT6_1_SOL),
    ],
)
async def test_an_llm_block_given_a_switched_off_model_runs_its_fallback(
    saved: str, runs: LLMModel
):
    assert await _run_ai_text_generator(saved) == [runs]


@pytest.mark.asyncio
async def test_an_llm_block_given_an_unknown_model_still_fails():
    with pytest.raises(BlockInputError):
        await _run_ai_text_generator("gpt-0")
