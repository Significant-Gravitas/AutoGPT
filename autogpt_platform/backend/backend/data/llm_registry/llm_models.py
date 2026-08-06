"""Model identity: the ``LLMModel`` enum and its catalog projections.

The catalog file (``catalog.py``) owns every model FACT; this module owns
the model NAMES — the stable identifiers block schemas serialize — plus the
projections built from the catalog at import (``MODEL_METADATA``, the
platform default, rename/alias maps). Lives in ``llm_registry`` so identity
and facts share one package; ``backend.blocks.llm`` re-exports everything
for the existing import surface.
"""

import logging
import re
from collections.abc import Mapping
from enum import Enum, EnumMeta
from typing import Literal, NamedTuple

from backend.data.llm_registry.catalog import get_catalog
from backend.data.llm_registry.catalog_model import CatalogPayload

logger = logging.getLogger(__name__)

# Anthropic snapshot-date suffix (claude-haiku-4-5-20251001 → -20251001).
# Shared by every slug canonicalizer so the pattern can't drift.
MODEL_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")


def transport_slug_candidates(slug: str) -> list[str]:
    """Spellings a transport-form slug may take in the catalog, in match
    order: exact, vendor-stripped, dots→dashes. ONE definition shared by
    every lookup that tolerates transport spellings (the copilot resolver
    today), so a spelling accepted in one path can't be refused in
    another. Enum resolution (``_missing_``/aliases) is deliberately
    separate — enum identity is exact-or-aliased, never fuzzy.
    """
    candidates = [slug]
    if "/" in slug:
        # ANY vendor prefix (anthropic/, openai/, …) may wrap a catalog
        # slug that is stored bare — exact match runs first, so slugs
        # that legitimately carry their prefix (moonshotai/…) still hit
        # before these fallbacks.
        tail = slug.split("/", 1)[1]
        candidates += [tail, tail.replace(".", "-")]
    elif slug.startswith("claude-"):
        candidates.append(slug.replace(".", "-"))
    return candidates


class ModelMetadata(NamedTuple):
    provider: str
    context_window: int
    max_output_tokens: int | None
    display_name: str
    provider_name: str
    creator_name: str
    price_tier: Literal[1, 2, 3]


class LLMModelMeta(EnumMeta):
    pass


class LLMModel(str, Enum, metaclass=LLMModelMeta):
    @classmethod
    def _missing_(cls, value: object) -> "LLMModel | None":
        """Resolve provider-prefixed model names.

        Handles two shapes:

        1. OpenRouter aliases for Anthropic models whose direct-API
           identifier carries a snapshot suffix that the OpenRouter slug
           drops — e.g. ``anthropic/claude-haiku-4-5`` ↔ enum value
           ``claude-haiku-4-5-20251001``.  Looked up via
           ``_OPENROUTER_ALIASES`` (defined below the class so it can hold
           ``LLMModel`` members directly).
        2. Generic provider prefix strip — e.g.
           ``anthropic/claude-sonnet-4-6`` → ``claude-sonnet-4-6``.
        """
        if not isinstance(value, str):
            return None
        alias = _OPENROUTER_ALIASES.get(value)
        if alias is not None:
            return alias
        if "/" in value:
            stripped = value.split("/", 1)[1]
            try:
                return cls(stripped)
            except ValueError:
                return None
        return None

    # OpenAI models
    O3_MINI = "o3-mini"
    O3 = "o3-2025-04-16"
    # GPT-5 models
    GPT5_2 = "gpt-5.2-2025-12-11"
    GPT5_1 = "gpt-5.1-2025-11-13"
    GPT5 = "gpt-5-2025-08-07"
    GPT5_MINI = "gpt-5-mini-2025-08-07"
    # O-series reasoning models
    O4_MINI = "o4-mini"
    O3_PRO = "o3-pro"
    O1 = "o1"
    O1_MINI = "o1-mini"
    # GPT-5.6 models (current flagship, July 2026)
    GPT5_6_SOL = "gpt-5.6-sol"
    GPT5_6_TERRA = "gpt-5.6-terra"
    GPT5_6_LUNA = "gpt-5.6-luna"
    # GPT-5.5 models
    GPT5_5 = "gpt-5.5-2026-04-23"
    GPT5_5_PRO = "gpt-5.5-pro"
    # GPT-5.4 models (March 2026)
    GPT5_4 = "gpt-5.4-2026-03-05"
    GPT5_4_MINI = "gpt-5.4-mini-2026-03-17"
    GPT5_4_NANO = "gpt-5.4-nano-2026-03-17"
    GPT5_4_PRO = "gpt-5.4-pro"
    # GPT-5.3 models
    GPT5_3 = "gpt-5.3-chat-latest"
    GPT5_3_CODEX = "gpt-5.3-codex"
    # Pro/codex variants of existing generations
    GPT5_2_PRO = "gpt-5.2-pro"
    GPT5_1_CODEX = "gpt-5.1-codex"
    GPT5_PRO = "gpt-5-pro"
    GPT41_NANO = "gpt-4.1-nano"
    GPT5_NANO = "gpt-5-nano-2025-08-07"
    GPT5_CHAT = "gpt-5-chat-latest"
    GPT41 = "gpt-4.1-2025-04-14"
    GPT41_MINI = "gpt-4.1-mini-2025-04-14"
    GPT4O_MINI = "gpt-4o-mini"
    GPT4O = "gpt-4o"
    # Anthropic models
    CLAUDE_4_5_OPUS = "claude-opus-4-5-20251101"
    CLAUDE_4_5_SONNET = "claude-sonnet-4-5-20250929"
    CLAUDE_4_5_HAIKU = "claude-haiku-4-5-20251001"
    CLAUDE_4_6_OPUS = "claude-opus-4-6"
    CLAUDE_4_7_OPUS = "claude-opus-4-7"
    CLAUDE_4_6_SONNET = "claude-sonnet-4-6"
    # AI/ML API models
    AIML_API_LLAMA3_3_70B = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    # Groq models
    LLAMA3_3_70B = "llama-3.3-70b-versatile"
    LLAMA3_1_8B = "llama-3.1-8b-instant"
    # Ollama models
    OLLAMA_LLAMA3_3 = "llama3.3"
    OLLAMA_LLAMA3_2 = "llama3.2"
    OLLAMA_LLAMA3_8B = "llama3"
    OLLAMA_LLAMA3_405B = "llama3.1:405b"
    OLLAMA_DOLPHIN = "dolphin-mistral:latest"
    # OpenRouter models
    OPENAI_GPT_OSS_120B = "openai/gpt-oss-120b"
    OPENAI_GPT_OSS_20B = "openai/gpt-oss-20b"
    GEMINI_2_5_PRO = "google/gemini-2.5-pro"
    GEMINI_3_1_PRO_PREVIEW = "google/gemini-3.1-pro-preview"
    GEMINI_3_FLASH_PREVIEW = "google/gemini-3-flash-preview"
    GEMINI_2_5_FLASH = "google/gemini-2.5-flash"
    GEMINI_2_0_FLASH = "google/gemini-2.0-flash-001"
    GEMINI_3_1_FLASH_LITE_PREVIEW = "google/gemini-3.1-flash-lite-preview"
    GEMINI_2_5_FLASH_LITE = "google/gemini-2.5-flash-lite"
    GEMINI_2_0_FLASH_LITE = "google/gemini-2.0-flash-lite-001"
    MISTRAL_LARGE_3 = "mistralai/mistral-large-2512"
    MISTRAL_MEDIUM_3_1 = "mistralai/mistral-medium-3.1"
    MISTRAL_SMALL_3_2 = "mistralai/mistral-small-3.2-24b-instruct"
    CODESTRAL = "mistralai/codestral-2508"
    COHERE_COMMAND_A_03_2025 = "cohere/command-a-03-2025"
    COHERE_COMMAND_A_TRANSLATE_08_2025 = "cohere/command-a-translate-08-2025"
    COHERE_COMMAND_A_REASONING_08_2025 = "cohere/command-a-reasoning-08-2025"
    COHERE_COMMAND_A_VISION_07_2025 = "cohere/command-a-vision-07-2025"
    DEEPSEEK_CHAT = "deepseek/deepseek-chat"  # Actually: DeepSeek V3
    DEEPSEEK_R1_0528 = "deepseek/deepseek-r1-0528"
    PERPLEXITY_SONAR = "perplexity/sonar"
    PERPLEXITY_SONAR_PRO = "perplexity/sonar-pro"
    PERPLEXITY_SONAR_REASONING_PRO = "perplexity/sonar-reasoning-pro"
    PERPLEXITY_SONAR_DEEP_RESEARCH = "perplexity/sonar-deep-research"
    NOUSRESEARCH_HERMES_3_LLAMA_3_1_405B = "nousresearch/hermes-3-llama-3.1-405b"
    NOUSRESEARCH_HERMES_3_LLAMA_3_1_70B = "nousresearch/hermes-3-llama-3.1-70b"
    AMAZON_NOVA_LITE_V1 = "amazon/nova-lite-v1"
    AMAZON_NOVA_MICRO_V1 = "amazon/nova-micro-v1"
    AMAZON_NOVA_PRO_V1 = "amazon/nova-pro-v1"
    MICROSOFT_PHI_4 = "microsoft/phi-4"
    GRYPHE_MYTHOMAX_L2_13B = "gryphe/mythomax-l2-13b"
    META_LLAMA_4_SCOUT = "meta-llama/llama-4-scout"
    META_LLAMA_4_MAVERICK = "meta-llama/llama-4-maverick"
    GROK_3 = "x-ai/grok-3"
    GROK_4 = "x-ai/grok-4"
    GROK_4_FAST = "x-ai/grok-4-fast"
    GROK_4_1_FAST = "x-ai/grok-4.1-fast"
    GROK_4_20 = "x-ai/grok-4.20"
    GROK_4_20_MULTI_AGENT = "x-ai/grok-4.20-multi-agent"
    GROK_CODE_FAST_1 = "x-ai/grok-code-fast-1"
    KIMI_K2_5 = "moonshotai/kimi-k2.5"
    KIMI_K2_6 = "moonshotai/kimi-k2.6"
    KIMI_K2_THINKING = "moonshotai/kimi-k2-thinking"
    KIMI_K3 = "moonshotai/kimi-k3"
    QWEN3_235B_A22B_THINKING = "qwen/qwen3-235b-a22b-thinking-2507"
    QWEN3_CODER = "qwen/qwen3-coder"
    # Z.ai (Zhipu) models
    ZAI_GLM_4_6 = "z-ai/glm-4.6"
    ZAI_GLM_4_6V = "z-ai/glm-4.6v"
    ZAI_GLM_4_7 = "z-ai/glm-4.7"
    ZAI_GLM_4_7_FLASH = "z-ai/glm-4.7-flash"
    ZAI_GLM_5 = "z-ai/glm-5"
    ZAI_GLM_5_TURBO = "z-ai/glm-5-turbo"
    ZAI_GLM_5V_TURBO = "z-ai/glm-5v-turbo"
    # Llama API models
    LLAMA_API_LLAMA_4_SCOUT = "Llama-4-Scout-17B-16E-Instruct-FP8"
    LLAMA_API_LLAMA4_MAVERICK = "Llama-4-Maverick-17B-128E-Instruct-FP8"
    LLAMA_API_LLAMA3_3_8B = "Llama-3.3-8B-Instruct"
    LLAMA_API_LLAMA3_3_70B = "Llama-3.3-70B-Instruct"
    # v0 by Vercel models
    V0_1_5_MD = "v0-1.5-md"
    V0_1_5_LG = "v0-1.5-lg"
    V0_1_0_MD = "v0-1.0-md"

    @classmethod
    def __get_pydantic_json_schema__(cls, schema, handler):
        json_schema = handler(schema)
        llm_model_metadata = {}
        for model in cls:
            # Kill-switched and non-GA models drop out of the picker
            # metadata but remain valid enum values — stored graphs
            # referencing them keep validating and executing.
            if model.value in _PICKER_HIDDEN_SLUGS:
                continue
            model_name = model.value
            metadata = model.metadata
            llm_model_metadata[model_name] = {
                "creator": metadata.creator_name,
                "creator_name": metadata.creator_name,
                "title": metadata.display_name,
                "provider": metadata.provider,
                "provider_name": metadata.provider_name,
                "name": model_name,
                "price_tier": metadata.price_tier,
            }
        json_schema["llm_model"] = True
        json_schema["llm_model_metadata"] = llm_model_metadata
        return json_schema

    @property
    def metadata(self) -> ModelMetadata:
        return MODEL_METADATA[self]

    @property
    def provider(self) -> str:
        return self.metadata.provider

    @property
    def context_window(self) -> int:
        return self.metadata.context_window

    @property
    def max_output_tokens(self) -> int | None:
        return self.metadata.max_output_tokens


# OpenRouter exposes Anthropic models under canonical ``anthropic/<model>``
# slugs that drop the snapshot-date suffix Anthropic's own API uses
# (``claude-haiku-4-5-20251001`` → ``anthropic/claude-haiku-4-5``). The
# generic provider-prefix strip in ``_missing_`` can't reverse the date
# truncation, so map the OpenRouter slugs to ``LLMModel`` members here.
# Only models whose canonical enum value carries a ``-YYYYMMDD`` snapshot
# suffix need entries; values without a snapshot (4.6/4.7+) are already
# covered by the prefix-strip path alone. Stored as ``LLMModel`` instances
# (not strings) so a rename or snapshot rotation on the enum follows the
# alias automatically — a stale entry becomes a load-time ``AttributeError``
# rather than a silent ``_missing_`` miss at runtime.
_OPENROUTER_ALIASES: Mapping[str, LLMModel] = {
    "anthropic/claude-haiku-4-5": LLMModel.CLAUDE_4_5_HAIKU,
    "anthropic/claude-opus-4-5": LLMModel.CLAUDE_4_5_OPUS,
    "anthropic/claude-sonnet-4-5": LLMModel.CLAUDE_4_5_SONNET,
    "openai/gpt-5.4": LLMModel.GPT5_4,
    "openai/gpt-5.4-mini": LLMModel.GPT5_4_MINI,
    "openai/gpt-5.4-nano": LLMModel.GPT5_4_NANO,
    "openai/gpt-5.5": LLMModel.GPT5_5,
}


def _build_model_metadata() -> dict["LLMModel", ModelMetadata]:
    """Project catalog facts into the block-facing metadata shape.

    The catalog file (``backend/data/llm_registry/catalog.py``) is the
    single source of truth for model facts; this module keeps only the
    ``LLMModel`` identifiers that block schemas serialize. Catalog models
    without an enum member (copilot-only models routed by slug) simply
    don't surface in blocks.
    """
    payload = get_catalog()
    providers = {p.name: p.display_name for p in payload.providers}
    creators = {c.name: c.display_name for c in payload.creators}
    members = {m.value: m for m in LLMModel}
    metadata: dict[LLMModel, ModelMetadata] = {}
    for model in payload.models:
        member = members.get(model.slug)
        if member is None:
            continue
        metadata[member] = ModelMetadata(
            provider=model.provider,
            context_window=model.context_window,
            max_output_tokens=model.max_output_tokens,
            display_name=model.display_name,
            provider_name=providers.get(model.provider, model.provider),
            creator_name=(
                creators.get(model.creator, "Unknown") if model.creator else "Unknown"
            ),
            price_tier=model.price_tier,
        )
    return metadata


MODEL_METADATA = _build_model_metadata()


def _picker_hidden_slugs(payload: "CatalogPayload") -> frozenset[str]:
    """Slugs hidden from the block picker: kill-switched models AND models
    not yet GA (EMPLOYEES/ADMINS/HIDDEN visibility) — the catalog's
    documented "who can SEE this" contract. Enum values stay valid either
    way, so stored graphs referencing a hidden model keep validating and
    executing.

    DECIDED SCOPE: at the block layer this filter is a UX control, not a
    safety control. A hand-crafted graph node can still select a
    kill-switched slug; execution of stored graphs must keep working after
    a kill (and keep billing), so the block layer deliberately does not
    veto. The hard-stop for an incident is the retirement CLI, which
    rewrites the nodes; copilot serving IS vetoed at resolution time.
    """
    return frozenset(
        m.slug for m in payload.models if not m.is_enabled or m.visibility != "GA"
    )


_PICKER_HIDDEN_SLUGS = _picker_hidden_slugs(get_catalog())


def _default_model_from_catalog() -> LLMModel:
    """The platform default IS the catalog's recommended model — one fact,
    one home. First enabled ``is_recommended`` entry with an enum identifier
    wins (catalog order); no recommendation is a data error caught at boot.

    The default must also be GA: it is offered to every user and must be
    picker-selectable, so it has to clear the same ``visibility == "GA"``
    bar as ``_PICKER_HIDDEN_SLUGS`` — otherwise the default could be a model
    the picker hides. A non-GA model flagged ``is_recommended`` is skipped
    here (pre-launch models are not the public default).
    """
    members = {m.value for m in LLMModel}
    first_enabled: LLMModel | None = None
    for model in get_catalog().models:
        if (
            not model.is_enabled
            or model.visibility != "GA"
            or model.slug not in members
        ):
            continue
        if model.is_recommended:
            return LLMModel(model.slug)
        if first_enabled is None:
            first_enabled = LLMModel(model.slug)
    # Killing the recommended model must not crash boot — fall back to the
    # first enabled block-selectable model (deterministic catalog order).
    if first_enabled is not None:
        logger.error(
            "catalog has no enabled is_recommended model — defaulting to %s",
            first_enabled.value,
        )
        return first_enabled
    raise ValueError("catalog declares no enabled block-selectable models")


DEFAULT_LLM_MODEL = _default_model_from_catalog()

# Family-aware mapping for legacy model values that have been retired from the
# `LLMModel` enum. Used by both the Prisma migration that rewrites stored graph
# definitions and by the boot-time safety net (`migrate_llm_models` in
# backend/data/graph.py) so a Claude Opus user lands on a newer Opus instead of
# the global GPT default. Keep this in sync with
# migrations/20260512120000_retire_deprecated_llm_models/migration.sql.
LEGACY_MODEL_MAPPINGS: dict[str, LLMModel] = {
    "claude-3-haiku-20240307": LLMModel.CLAUDE_4_5_HAIKU,
    "claude-opus-4-20250514": LLMModel.CLAUDE_4_7_OPUS,
    "claude-sonnet-4-20250514": LLMModel.CLAUDE_4_6_SONNET,
    "claude-opus-4-1-20250805": LLMModel.CLAUDE_4_7_OPUS,
    "gpt-4-turbo": LLMModel.GPT41,
    "o1": LLMModel.O3,
    "o1-mini": LLMModel.O3_MINI,
    "google/gemini-2.5-pro-preview-03-25": LLMModel.GEMINI_2_5_PRO,
    "google/gemini-2.5-flash-lite-preview-06-17": LLMModel.GEMINI_2_5_FLASH,
    "cohere/command-r-08-2024": LLMModel.COHERE_COMMAND_A_03_2025,
    "cohere/command-r-plus-08-2024": LLMModel.COHERE_COMMAND_A_03_2025,
    "mistralai/mistral-nemo": LLMModel.MISTRAL_SMALL_3_2,
    "microsoft/wizardlm-2-8x22b": LLMModel.MICROSOFT_PHI_4,
    "moonshotai/kimi-k2": LLMModel.KIMI_K2_6,
    "moonshotai/kimi-k2-0905": LLMModel.KIMI_K2_6,
    "z-ai/glm-4-32b": LLMModel.ZAI_GLM_4_6,
    "z-ai/glm-4.5": LLMModel.ZAI_GLM_4_6,
    "z-ai/glm-4.5-air": LLMModel.ZAI_GLM_4_7_FLASH,
    "z-ai/glm-4.5-air:free": LLMModel.ZAI_GLM_4_7_FLASH,
    "z-ai/glm-4.5v": LLMModel.ZAI_GLM_4_6V,
    # AI/ML API stragglers — no direct same-family successor on AI/ML's current
    # catalogue, so they all map to the closest open-weight Meta/Llama option
    # that AI/ML still serves.
    "Qwen/Qwen2.5-72B-Instruct-Turbo": LLMModel.AIML_API_LLAMA3_3_70B,
    "nvidia/llama-3.1-nemotron-70b-instruct": LLMModel.AIML_API_LLAMA3_3_70B,
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": LLMModel.AIML_API_LLAMA3_3_70B,
    "meta-llama/Llama-3.2-3B-Instruct-Turbo": LLMModel.AIML_API_LLAMA3_3_70B,
}


def _assert_metadata_complete() -> None:
    for member in LLMModel:
        if member not in MODEL_METADATA:
            raise ValueError(f"Missing MODEL_METADATA metadata for model: {member}")


_assert_metadata_complete()
