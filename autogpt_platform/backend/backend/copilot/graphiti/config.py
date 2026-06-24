"""Configuration for Graphiti temporal knowledge graph integration."""

import os
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from backend.util.clients import OPENROUTER_BASE_URL

_BACKEND_ROOT = Path(__file__).resolve().parents[3]


# Cloud defaults that ``_apply_local_graphiti_models`` rewrites under
# local transport. Pinned as module constants so the field defaults and
# the validator can't drift — same pattern as
# ``copilot/config.py::_DEFAULT_TITLE_MODEL`` etc.
_DEFAULT_LLM_MODEL = "gpt-4.1-mini"
_DEFAULT_RERANKER_MODEL = "gpt-4.1-nano"
_DEFAULT_EMBEDDER_MODEL = "text-embedding-3-small"

# Local-transport defaults. Mirrors dev's chat-side ``--with-ollama``
# default (``hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M``, per
# ``docs/platform/copilot-local-llm.md`` — 4B params, ~3.4GB resident,
# 256k native context, vetted for OpenAI-shim structured output). The
# reranker reuses the same model since the prompts are simpler than
# extraction and pulling a second model for it would double the local
# install's disk + RAM footprint.
_LOCAL_LLM_MODEL = "hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M"
_LOCAL_RERANKER_MODEL = _LOCAL_LLM_MODEL
# Embedder defaults to Ollama's ``nomic-embed-text`` (~270MB, 768-dim
# vectors, well-supported by the OpenAI ``/v1/embeddings`` shim).
# Operators on ``--with-ollama-embeddings`` get this pulled
# automatically; everyone else needs to ``ollama pull nomic-embed-text``
# before graphiti ingest starts working — covered in copilot-local-llm.md.
_LOCAL_EMBEDDER_MODEL = "nomic-embed-text"


class GraphitiConfig(BaseSettings):
    """Configuration for Graphiti memory integration.

    All fields use the ``GRAPHITI_`` env-var prefix, e.g. ``GRAPHITI_ENABLED``.
    LLM/embedder keys fall back to the AutoPilot-dedicated keys
    (``CHAT_API_KEY`` / ``CHAT_OPENAI_API_KEY``) so that memory costs are
    tracked under AutoPilot, then to the platform-wide OpenRouter / OpenAI
    keys as a last resort.
    """

    model_config = SettingsConfigDict(env_prefix="GRAPHITI_", extra="allow")

    # FalkorDB connection
    falkordb_host: str = Field(default="localhost")
    falkordb_port: int = Field(default=6380)
    falkordb_password: str = Field(default="")

    # LLM for entity extraction (used by graphiti-core during ingestion).
    # Default is a cloud OpenAI-compat slug. Under ``CHAT_USE_LOCAL=true``
    # ``_apply_local_graphiti_models`` rewrites this to the same Ollama
    # default that dev's PR #12993 uses for the chat path so a single
    # ``ollama pull`` powers both surfaces.
    llm_model: str = Field(
        default=_DEFAULT_LLM_MODEL,
        description="Model for entity extraction — must support structured output",
    )
    llm_base_url: str = Field(
        default="",
        description="Base URL for LLM API — empty falls back to OPENROUTER_BASE_URL",
    )
    llm_api_key: str = Field(
        default="",
        description="API key for LLM — empty falls back to CHAT_API_KEY, then OPEN_ROUTER_API_KEY",
    )

    # Embedder (separate from LLM — embeddings go direct to OpenAI).
    # Under local transport rewritten to ``nomic-embed-text`` (Ollama).
    embedder_model: str = Field(default=_DEFAULT_EMBEDDER_MODEL)
    embedder_base_url: str = Field(
        default="",
        description="Base URL for embedder — empty uses OpenAI direct",
    )
    embedder_api_key: str = Field(
        default="",
        description="API key for embedder — empty falls back to CHAT_OPENAI_API_KEY, then OPENAI_API_KEY",
    )

    # Cross-encoder reranker (P-1.4) — used by warm-context retrieval to
    # rerank top edges from BM25 + cosine + BFS. Graphiti's built-in
    # OpenAIRerankerClient runs concurrent boolean-classifier prompts
    # against gpt-4.1-nano by default (one prompt per candidate; log-
    # probabilities decide the score). The audit estimated ~10–15%
    # precision lift on warm context at the cost of one LLM call per
    # session start. Defaults match Graphiti's own default so the
    # reranker can ship with no env config.
    reranker_model: str = Field(
        default=_DEFAULT_RERANKER_MODEL,
        description="Model for the cross-encoder reranker. Cheap, fast classifier prompts.",
    )

    # Activity-gate threshold for community rebuilds (P-1.7).
    # ``rebuild_communities_for_user`` skips when there have been fewer
    # than this many new episodes since the last successful rebuild —
    # avoids paying the per-community LLM-summarization cost AND avoids
    # clustering drift on essentially-unchanged graphs (LP tie-breaks
    # are non-deterministic; summary text varies). Defaults to 5 — a
    # full week of low-activity (~1 episode/day) is fine to skip; a
    # power user blasting 20+ memories triggers rebuild within hours.
    community_rebuild_min_new_episodes: int = Field(
        default=5,
        description="Skip community rebuild when fewer than this many new episodes since last rebuild.",
    )

    # Nightly community rebuild is non-interactive scheduled work — ~50%
    # cheaper on OpenAI's ``service_tier="flex"`` (worst-case ~15min
    # queue, typically just a few seconds slower). Interactive ingest
    # stays on sync tier because the user expects deduped facts ready
    # for their next turn.
    community_rebuild_use_flex_tier: bool = Field(
        default=True,
        description="Run nightly community rebuilds on OpenAI's flex service tier (~50% discount).",
    )

    # Concurrency
    semaphore_limit: int = Field(
        default=5,
        description="Max concurrent LLM calls during ingestion (prevents rate limits)",
    )

    # Warm context
    context_max_facts: int = Field(default=20)
    context_timeout: float = Field(
        default=8.0,
        description="Seconds before warm context fetch is abandoned (needs headroom for FalkorDB cold connections)",
    )

    # Client cache
    client_cache_maxsize: int = Field(default=500)
    client_cache_ttl: int = Field(
        default=1800,
        description="TTL in seconds for cached Graphiti client instances (30 min)",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            file_secret_settings,
            DotEnvSettingsSource(settings_cls, env_file=_BACKEND_ROOT / ".env"),
            DotEnvSettingsSource(settings_cls, env_file=_BACKEND_ROOT / ".env.default"),
        )

    def _chat_local_fallback(self) -> tuple[str, str] | None:
        """Return ``(api_key, base_url)`` when the active chat transport
        is ``local`` — otherwise ``None``.

        Holy-grail integration with dev's ``TransportProfile`` from
        PR #12993: under ``CHAT_USE_LOCAL=true``, graphiti's entity
        extraction + embedder need to hit the same self-hosted backend
        that the chat layer was pointed at. The cloud-key fallback chain
        (CHAT_API_KEY → OPEN_ROUTER_API_KEY) returns either nothing or
        a stray cloud key the operator forgot to unset, which would
        send a request to the wrong endpoint and 401.

        Lazy-imports ``chat_cfg`` to keep the
        ``copilot.sdk.env`` ↔ ``util/clients`` import cycle dev
        already established under control. Returns ``None`` for any
        non-local transport so the caller falls through to its
        existing cloud-key chain (cloud users are unaffected).
        """
        # Local import: ``copilot.sdk.env`` instantiates a ``ChatConfig``
        # at import time, and graphiti's config is sometimes imported
        # at module-collection time (CI, tools) before env is fully
        # primed. Lazy so the import only fires when a resolver is
        # actually called.
        from backend.copilot.sdk.env import config as chat_cfg

        if chat_cfg.transport.name != "local":
            return None
        return chat_cfg.api_key or "", chat_cfg.base_url or ""

    def resolve_llm_api_key(self) -> str:
        if self.llm_api_key:
            return self.llm_api_key
        # Prefer the AutoPilot-dedicated key so memory costs are tracked
        # separately from the platform-wide OpenRouter key.
        cloud_key = os.getenv("CHAT_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY")
        if cloud_key:
            return cloud_key
        # Last resort: under local transport, inherit the placeholder
        # key the operator set for the chat path so the Ollama backend
        # accepts the bearer (any non-empty value).
        local_fallback = self._chat_local_fallback()
        if local_fallback is not None:
            return local_fallback[0]
        return ""

    def resolve_llm_base_url(self) -> str:
        if self.llm_base_url:
            return self.llm_base_url
        # Under local transport, point at the same OpenAI-compat
        # endpoint as the chat layer (e.g. http://localhost:11434/v1).
        # Cloud transports keep the OpenRouter default.
        local_fallback = self._chat_local_fallback()
        if local_fallback is not None and local_fallback[1]:
            return local_fallback[1]
        return OPENROUTER_BASE_URL

    def resolve_embedder_api_key(self) -> str:
        if self.embedder_api_key:
            return self.embedder_api_key
        # Prefer the AutoPilot-dedicated OpenAI key so memory costs are
        # tracked separately from the platform-wide OpenAI key.
        cloud_key = os.getenv("CHAT_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if cloud_key:
            return cloud_key
        # Last resort: under local transport, inherit the chat placeholder.
        local_fallback = self._chat_local_fallback()
        if local_fallback is not None:
            return local_fallback[0]
        return ""

    def resolve_embedder_base_url(self) -> str | None:
        if self.embedder_base_url:
            return self.embedder_base_url
        # Under local transport, embeddings go to the same self-hosted
        # backend (the operator pulled an embedding model into Ollama
        # per docs/platform/copilot-local-llm.md). Cloud transports use
        # the OpenAI SDK default (``None``).
        local_fallback = self._chat_local_fallback()
        if local_fallback is not None and local_fallback[1]:
            return local_fallback[1]
        return None  # OpenAI SDK default

    @model_validator(mode="after")
    def _apply_local_graphiti_models(self) -> "GraphitiConfig":
        """Rewrite cloud model defaults to local-friendly slugs under
        ``CHAT_USE_LOCAL=true``.

        Mirror of dev's ``_apply_local_aux_models`` pattern on
        ``ChatConfig`` (PR #12993). Three slots get rewritten when
        still at their cloud defaults and the operator hasn't pinned
        a ``GRAPHITI_*_MODEL`` override:

        - ``llm_model`` (``gpt-4.1-mini``) → ``hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M``
          (matches dev's chat default — one Ollama pull powers both
          surfaces; vetted for structured output / tool-calling shape).
        - ``reranker_model`` (``gpt-4.1-nano``) → same Qwen slug.
          Reranker prompts are simpler than extraction; reusing the
          chat model avoids pulling a second model just for reranking.
        - ``embedder_model`` (``text-embedding-3-small``) →
          ``nomic-embed-text``. Operators need to ``ollama pull
          nomic-embed-text`` separately (chat doesn't need it, so the
          installer's ``--with-ollama`` doesn't cover it today; see
          docs/platform/copilot-local-llm.md).

        Operator overrides (custom slugs like ``qwen3:8b`` or
        ``hf.co/...``) pass through untouched — only the literal cloud
        defaults trigger rewriting. Lazy-imports ``chat_cfg`` to avoid
        the ``copilot.sdk.env`` ↔ ``util/clients`` cycle dev's PR
        established.
        """
        try:
            from backend.copilot.sdk.env import config as chat_cfg
        except Exception:
            # Defensive: if the chat config fails to import for any
            # reason (corrupt env vars during test collection, etc.)
            # keep the cloud defaults rather than crashing graphiti
            # at import time. Operators on local transport see the
            # cloud default and a clear 404 from Ollama; everyone
            # else is unaffected.
            return self

        if chat_cfg.transport.name != "local":
            return self

        if self.llm_model == _DEFAULT_LLM_MODEL:
            object.__setattr__(self, "llm_model", _LOCAL_LLM_MODEL)
        if self.reranker_model == _DEFAULT_RERANKER_MODEL:
            object.__setattr__(self, "reranker_model", _LOCAL_RERANKER_MODEL)
        if self.embedder_model == _DEFAULT_EMBEDDER_MODEL:
            object.__setattr__(self, "embedder_model", _LOCAL_EMBEDDER_MODEL)
        return self


_graphiti_config: GraphitiConfig | None = None


def _get_config() -> GraphitiConfig:
    global _graphiti_config
    if _graphiti_config is None:
        _graphiti_config = GraphitiConfig()
    return _graphiti_config


# Backwards-compatible module-level attribute access.
# All internal code should use ``_get_config()`` to avoid import-time
# construction, but this keeps existing ``graphiti_config.xxx`` usage working.
class _LazyConfigProxy:
    def __getattr__(self, name: str):
        return getattr(_get_config(), name)


graphiti_config = _LazyConfigProxy()  # type: ignore[assignment]


async def is_enabled_for_user(user_id: str | None) -> bool:
    """Check if Graphiti memory is enabled for a specific user.

    Gated solely by LaunchDarkly flag ``graphiti-memory``
    (Flag.GRAPHITI_MEMORY).  When LD is not configured, defaults to False.
    """
    if not user_id:
        return False

    from backend.util.feature_flag import Flag, is_feature_enabled

    return await is_feature_enabled(
        Flag.GRAPHITI_MEMORY,
        user_id,
        default=False,
    )


async def is_communities_enabled_for_user(user_id: str | None) -> bool:
    """Check if per-user community-detection rebuilds are enabled.

    Distinct from ``is_enabled_for_user`` — a user can have Graphiti
    memory enabled (writes + reads work) without the weekly Leiden
    rebuild + LLM summarization running on their graph. Gated by
    ``Flag.GRAPHITI_COMMUNITIES_ENABLED``; defaults False so the cost
    only lands behind explicit opt-in.
    """
    if not user_id:
        return False

    from backend.util.feature_flag import Flag, is_feature_enabled

    return await is_feature_enabled(
        Flag.GRAPHITI_COMMUNITIES_ENABLED,
        user_id,
        default=False,
    )
