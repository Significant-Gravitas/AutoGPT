"""Every LLM model value stored in real AgentNode data resolves via the catalog.

Fixture captured 2026-07-20 with a read-only query against BOTH prod and dev:

    SELECT DISTINCT "constantInput"->>'model' FROM "AgentNode";

Saved graphs keep the model as a plain string in ``constantInput`` forever —
they are a fossil record of every model identifier we ever shipped, so code
inspection alone cannot prove that cutting model resolution over to the
catalog strands no existing graph. This fixture makes that set closed and
testable: every stored value must either resolve through the catalog
(directly or via ``LEGACY_MODEL_MAPPINGS``) or be claimed by a non-LLM block
(image/video/search/embedding blocks share the ``model`` input name).

Re-run the query and refresh ``STORED_MODEL_VALUES`` before any change that
narrows model resolution (model removal, legacy-mapping cleanup).
"""

from __future__ import annotations

from pathlib import Path

from backend.blocks.llm import LEGACY_MODEL_MAPPINGS, LLMModel
from backend.data.llm_registry.catalog import get_catalog

# Merged prod + dev distinct values, 2026-07-20. NULL rows (nodes without a
# ``model`` key) excluded — absent keys fall back to the block default.
STORED_MODEL_VALUES = [
    "Flux 1.1 Pro",
    "Flux 1.1 Pro Ultra",
    "Flux Kontext Max",
    "Flux Kontext Pro",
    "Llama-3.3-70B-Instruct",
    "Llama-3.3-8B-Instruct",
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Llama-4-Scout-17B-16E-Instruct-FP8",
    "Nano Banana 2",
    "Nano Banana Pro",
    "Recraft v3",
    "Stable Diffusion 3.5 Medium",
    "amazon/nova-lite-v1",
    "amazon/nova-micro-v1",
    "amazon/nova-pro-v1",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-5-20251101",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-6",
    "cohere/command-a-03-2025",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-r1-0528",
    "dolphin-mistral:latest",
    "exa",
    "exa-research",
    "exa-research-pro",
    "fal-ai/luma-dream-machine",
    "fal-ai/mochi-v1",
    "fal-ai/veo3",
    "google/gemini-2.0-flash-001",
    "google/gemini-2.0-flash-lite-001",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-image",
    "google/gemini-2.5-pro",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-flash-lite-preview",
    "google/gemini-3.1-pro-preview",
    "google/nano-banana",
    "google/nano-banana-2",
    "google/nano-banana-pro",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5-2025-08-07",
    "gpt-5-chat-latest",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07",
    "gpt-5.1-2025-11-13",
    "gpt-5.1-codex",
    "gpt-5.2-2025-12-11",
    "gryphe/mythomax-l2-13b",
    "jina-embeddings-v2-base-en",
    "kontext-pro",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3",
    "llama3.1:405b",
    "llama3.2",
    "llama3.3",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-4-scout",
    "microsoft/phi-4",
    "mistralai/codestral-2508",
    "mistralai/mistral-large-2512",
    "mistralai/mistral-medium-3.1",
    "mistralai/mistral-small-3.2-24b-instruct",
    "moonshotai/kimi-k2.5",
    "moonshotai/kimi-k2.6",
    "nano-banana-pro",
    "nousresearch/hermes-3-llama-3.1-405b",
    "nousresearch/hermes-3-llama-3.1-70b",
    "o3-2025-04-16",
    "o3-mini",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "perplexity/sonar",
    "perplexity/sonar-deep-research",
    "perplexity/sonar-pro",
    "qwen/qwen3-235b-a22b-thinking-2507",
    "qwen/qwen3-coder",
    "v0-1.0-md",
    "v0-1.5-lg",
    "v0-1.5-md",
    "x-ai/grok-4",
    "x-ai/grok-4.1-fast",
    "x-ai/grok-4.20-multi-agent",
    "x-ai/grok-code-fast-1",
    "z-ai/glm-5v-turbo",
]

# Stored values owned by non-LLM blocks (they also name their input ``model``),
# mapped to the block module that serves them. Values marked historical no
# longer appear in the owning block's current enum; their migration story
# belongs to that block, not the LLM catalog.
NON_LLM_BLOCK_VALUES: dict[str, str] = {
    "Flux 1.1 Pro": "ai_image_generator_block.py",
    "Flux 1.1 Pro Ultra": "ai_image_generator_block.py",
    "Flux Kontext Max": "flux_kontext.py",
    "Flux Kontext Pro": "flux_kontext.py",
    "Nano Banana 2": "ai_image_generator_block.py",
    "Nano Banana Pro": "ai_image_generator_block.py",
    "Recraft v3": "ai_image_generator_block.py",
    "Stable Diffusion 3.5 Medium": "ai_image_generator_block.py",
    "exa": "exa/research.py",  # historical
    "exa-research": "exa/research.py",
    "exa-research-pro": "exa/research.py",
    "fal-ai/luma-dream-machine": "fal/ai_video_generator.py",
    "fal-ai/mochi-v1": "fal/ai_video_generator.py",
    "fal-ai/veo3": "fal/ai_video_generator.py",
    "google/gemini-2.5-flash-image": "ai_image_generator_block.py",  # historical
    "google/nano-banana": "ai_image_customizer.py",
    "google/nano-banana-2": "ai_image_generator_block.py",
    "google/nano-banana-pro": "ai_image_generator_block.py",
    "gpt-5.1-codex": "codex.py",
    "jina-embeddings-v2-base-en": "jina/embeddings.py",
    "kontext-pro": "flux_kontext.py",
    "nano-banana-pro": "ai_image_customizer.py",
}

BLOCKS_DIR = Path(__file__).parents[2] / "blocks"


def test_every_stored_llm_model_value_resolves_through_the_catalog():
    # Block runtime resolves via the ENUM, whose identifiers must be catalog-
    # backed (import-time check) — assert against enum values so a future
    # copilot-only catalog model can't mask a value blocks would reject.
    enum_values = {m.value for m in LLMModel}
    catalog_slugs = {m.slug for m in get_catalog().models}
    assert enum_values <= catalog_slugs
    unresolved = []
    for value in STORED_MODEL_VALUES:
        if value in NON_LLM_BLOCK_VALUES:
            continue
        if value in enum_values:
            continue
        legacy = LEGACY_MODEL_MAPPINGS.get(value)
        if legacy is not None and legacy.value in enum_values:
            continue
        unresolved.append(value)
    assert not unresolved, (
        "Model values stored in real graphs would no longer resolve after the "
        f"catalog cutover: {unresolved}. Add the model to the catalog or map "
        "it in LEGACY_MODEL_MAPPINGS before shipping."
    )


def test_non_llm_exclusions_name_real_block_modules():
    missing = {
        value: module
        for value, module in NON_LLM_BLOCK_VALUES.items()
        if not (BLOCKS_DIR / module).exists()
    }
    assert (
        not missing
    ), f"Exclusion entries point at block modules that do not exist: {missing}"


def test_exclusion_list_stays_scoped_to_the_fixture():
    stale = set(NON_LLM_BLOCK_VALUES) - set(STORED_MODEL_VALUES)
    assert not stale, f"Exclusions for values no longer present in the fixture: {stale}"
