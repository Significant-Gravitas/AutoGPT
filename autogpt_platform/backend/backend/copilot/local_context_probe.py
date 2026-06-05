"""Auto-detect the context window of a local OpenAI-compatible LLM backend.

The local transport (``CHAT_USE_LOCAL=true``) talks to an operator-run backend
(Ollama, vLLM, LM Studio, llama.cpp server, …). That backend's *loaded* context
window — not any AutoGPT-side config — is the real ceiling AutoPilot must compact
its conversation under. Rather than carry a second config value that has to be
kept in lockstep with the server (and silently drifts), we read the window back
from the backend at runtime.

Probe order (first hit wins; every probe is best-effort, all errors swallowed):

    Ollama          GET {root}/api/ps              models[].context_length
    llama.cpp       GET {root}/props              default_generation_settings.n_ctx
    vLLM            GET {base}/models             data[].max_model_len
    LM Studio       GET {root}/api/v0/models      data[].max_context_length

Backends that expose nothing standard (LiteLLM proxy, Jan, text-generation-webui)
fall back to ``LOCAL_CONTEXT_FALLBACK`` (32768) — conservative: AutoPilot compacts
as if the window were 32k, retaining less history rather than overflowing.

``{root}`` is the base URL minus a trailing ``/v1``. The Ollama probe only returns
a value while a model is loaded; that's fine because the window is needed for
*compaction*, which only matters after a few turns — by which point an earlier
turn has already loaded the model.
"""

import logging
import time

import httpx

logger = logging.getLogger(__name__)

# ``compaction_target_for_window`` reserves this much of the window for the
# static per-turn floor (system prompt + ~43 tool schemas ≈ 19k measured) plus
# ~5k headroom for content a turn appends *after* the turn-start compaction
# check (tool results, the new user message) — without it a chunky tool result
# can push one turn past a small window before the next compaction fires.
# ``compress_context`` applies its own ~2k response reserve on top. The
# conversation-history budget is therefore ``window - _FLOOR_RESERVE``.
_FLOOR_RESERVE = 24_000
_TARGET_FLOOR = 4_096

# Used when no backend reports a window. Matches the installer's default
# ``OLLAMA_CONTEXT_LENGTH`` so the common Ollama path is correct even when the
# model is briefly unloaded and ``/api/ps`` is empty.
LOCAL_CONTEXT_FALLBACK = 32_768

# Below this, the ~19k floor leaves almost no room for conversation — the
# operator's backend window is misconfigured for AutoPilot.
_MINIMUM_SAFE_WINDOW = 24_576

_PROBE_TIMEOUT_S = 2.0
_CACHE_TTL_S = 300.0

# Both keyed by (base_url, model): vLLM / LM Studio expose a *per-model* window,
# so caching by base_url alone would serve one model's window for another.
# ``_last_window`` never expires and is reused when a probe can't determine the
# window (e.g. the model isn't loaded yet) — a far better fallback than the
# optimistic constant.
_CacheKey = tuple[str, str]
_probe_cache: dict[_CacheKey, tuple[int, float]] = {}
_last_window: dict[_CacheKey, int] = {}


def compaction_target_for_window(window: int) -> int:
    """Token budget for conversation history given the backend's ``window``.

    Floored at ``_TARGET_FLOOR`` so the value is never zero/negative on a
    pathologically small window (a separate WARNING is logged by the probe).
    """
    return max(_TARGET_FLOOR, window - _FLOOR_RESERVE)


def _cache_get(key: _CacheKey) -> int | None:
    entry = _probe_cache.get(key)
    if entry is None:
        return None
    window, fetched_at = entry
    if time.monotonic() - fetched_at > _CACHE_TTL_S:
        del _probe_cache[key]
        return None
    return window


def _server_root(base_url: str) -> str:
    """Strip a trailing ``/v1`` (or ``/v1/``) from the OpenAI-compat base URL."""
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url.rstrip("/")


async def probe_local_context_window(base_url: str, model: str) -> int:
    """Return the loaded context window (tokens) for ``model`` at ``base_url``.

    Cached per ``(base_url, model)`` for 5 minutes so it never fires on every
    turn. When a probe can't determine the window (e.g. the model isn't loaded
    yet), reuses the last successfully-detected window for that endpoint+model
    if known — a far better estimate than the optimistic constant — and does
    NOT cache the miss, so the next turn re-probes once a model is loaded. Logs
    a WARNING when a detected window is below ``_MINIMUM_SAFE_WINDOW``.
    """
    key = (base_url, model)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    detected = await _detect_window(base_url, model)
    if detected is None:
        return _last_window.get(key, LOCAL_CONTEXT_FALLBACK)

    _last_window[key] = detected
    if detected < _MINIMUM_SAFE_WINDOW:
        logger.warning(
            "[LocalProbe] Backend at %s reports a %d-token context window — below "
            "the %d-token minimum AutoPilot needs (its system prompt + tools use "
            "~19k alone, leaving only ~%d for conversation). Raise the backend's "
            "context length (e.g. OLLAMA_CONTEXT_LENGTH=%d) to avoid truncation.",
            base_url,
            detected,
            _MINIMUM_SAFE_WINDOW,
            max(0, detected - _FLOOR_RESERVE),
            _MINIMUM_SAFE_WINDOW,
        )
    _probe_cache[key] = (detected, time.monotonic())
    return detected


async def _detect_window(base_url: str, model: str) -> int | None:
    root = _server_root(base_url)
    base = base_url.rstrip("/")
    model_base = model.split(":")[0]

    async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S) as client:
        # 1. Ollama — GET {root}/api/ps -> models[].context_length.
        # The window is a server-wide setting (OLLAMA_CONTEXT_LENGTH), so any
        # loaded model reflects it; prefer a name match, else the first model.
        try:
            resp = await client.get(f"{root}/api/ps")
            if resp.status_code == 200:
                models = resp.json().get("models") or []
                match = next(
                    (
                        m
                        for m in models
                        if str(m.get("name", "")).startswith(model_base)
                    ),
                    models[0] if models else None,
                )
                if match:
                    ctx = match.get("context_length")
                    if isinstance(ctx, int) and ctx > 0:
                        logger.debug(
                            "[LocalProbe] Ollama /api/ps context_length=%d", ctx
                        )
                        return ctx
        except Exception as exc:
            logger.debug("[LocalProbe] /api/ps probe failed: %s", exc)

        # 2. llama.cpp server — GET {root}/props -> default_generation_settings.n_ctx
        try:
            resp = await client.get(f"{root}/props")
            if resp.status_code == 200:
                n_ctx = (resp.json().get("default_generation_settings") or {}).get(
                    "n_ctx"
                )
                if isinstance(n_ctx, int) and n_ctx > 0:
                    logger.debug("[LocalProbe] llama.cpp /props n_ctx=%d", n_ctx)
                    return n_ctx
        except Exception as exc:
            logger.debug("[LocalProbe] /props probe failed: %s", exc)

        # 3. vLLM — GET {base}/models -> data[].max_model_len
        try:
            resp = await client.get(f"{base}/models")
            if resp.status_code == 200:
                for m in resp.json().get("data") or []:
                    if model in (m.get("id"), m.get("name")):
                        ctx = m.get("max_model_len")
                        if isinstance(ctx, int) and ctx > 0:
                            logger.debug("[LocalProbe] vLLM max_model_len=%d", ctx)
                            return ctx
        except Exception as exc:
            logger.debug("[LocalProbe] /models probe failed: %s", exc)

        # 4. LM Studio — GET {root}/api/v0/models -> data[].max_context_length
        try:
            resp = await client.get(f"{root}/api/v0/models")
            if resp.status_code == 200:
                for m in resp.json().get("data") or []:
                    if model in (m.get("id"), m.get("name")):
                        ctx = m.get("max_context_length")
                        if isinstance(ctx, int) and ctx > 0:
                            logger.debug(
                                "[LocalProbe] LM Studio max_context_length=%d", ctx
                            )
                            return ctx
        except Exception as exc:
            logger.debug("[LocalProbe] /api/v0/models probe failed: %s", exc)

    logger.debug(
        "[LocalProbe] No backend window detected at %s; caller will fall back",
        base_url,
    )
    return None
