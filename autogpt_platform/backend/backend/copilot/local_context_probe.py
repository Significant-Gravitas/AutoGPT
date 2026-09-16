"""Auto-detect the context window of a local OpenAI-compatible LLM backend.

The local transport (``CHAT_USE_LOCAL=true``) talks to an operator-run backend
(Ollama, vLLM, LM Studio, llama.cpp server, …). That backend's *loaded* context
window — not any AutoGPT-side config — is the real ceiling Otto must compact
its conversation under. Rather than carry a second config value that has to be
kept in lockstep with the server (and silently drifts), we read the window back
from the backend at runtime.

Probe order (first hit wins; every probe is best-effort, all errors swallowed):

    Ollama          GET {root}/api/ps              models[].context_length
    llama.cpp       GET {root}/props              default_generation_settings.n_ctx
    vLLM            GET {base}/models             data[].max_model_len
    LM Studio       GET {root}/api/v0/models      data[].max_context_length

Backends that expose nothing standard (LiteLLM proxy, Jan, text-generation-webui)
fall back to ``LOCAL_CONTEXT_FALLBACK`` (32768) — conservative: Otto compacts
as if the window were 32k, retaining less history rather than overflowing.

``{root}`` is the base URL minus a trailing ``/v1``. The Ollama probe only returns
a value while a model is loaded; that's fine because the window is needed for
*compaction*, which only matters after a few turns — by which point an earlier
turn has already loaded the model.
"""

import logging
import time
from dataclasses import dataclass

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

# Used when no backend reports a window. Deliberately remains 32k even though
# the default Ornith installer uses 262k: assuming a huge window for a backend
# that cannot report one risks overflowing its actual context limit.
LOCAL_CONTEXT_FALLBACK = 32_768

# Below this, the ~19k floor leaves almost no room for conversation — the
# operator's backend window is misconfigured for Otto.
_MINIMUM_SAFE_WINDOW = 24_576

# Below this, the SDK CLI cannot run at all: its fixed floor of system
# prompt plus tool definitions measures ~65-110k tokens, so a backend
# reporting less has no room even before any conversation. Only enforced
# against windows a backend positively reports — an unknown window (no
# report, nothing remembered) proceeds on the fallback instead, because
# failing closed there would deadlock first turns: the turn itself is
# what loads the model.
SDK_MINIMUM_CONTEXT_WINDOW = 65_000

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


@dataclass(frozen=True)
class LocalWindowProbe:
    """Probed local-backend window plus its provenance.

    ``detected`` is True when a backend positively reported the window
    (freshly, from cache, or remembered from an earlier turn) and False
    when it is the blind ``LOCAL_CONTEXT_FALLBACK``. The SDK floor guard
    only fires on detected windows — failing closed on unknown ones
    would deadlock first turns (see ``SDK_MINIMUM_CONTEXT_WINDOW``).
    """

    window: int
    detected: bool


async def probe_local_context_window_status(
    base_url: str, model: str
) -> LocalWindowProbe:
    """Like :func:`probe_local_context_window`, plus provenance.

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
        return LocalWindowProbe(window=cached, detected=True)

    detected = await _detect_window(base_url, model)
    if detected is None:
        if key in _last_window:
            return LocalWindowProbe(window=_last_window[key], detected=True)
        return LocalWindowProbe(window=LOCAL_CONTEXT_FALLBACK, detected=False)

    _last_window[key] = detected
    if detected < _MINIMUM_SAFE_WINDOW:
        logger.warning(
            "[LocalProbe] Backend at %s reports a %d-token context window — below "
            "the %d-token minimum Otto needs (its system prompt + tools use "
            "~19k alone, leaving only ~%d for conversation). Raise the backend's "
            "context length (e.g. OLLAMA_CONTEXT_LENGTH=%d) to avoid truncation.",
            base_url,
            detected,
            _MINIMUM_SAFE_WINDOW,
            max(0, detected - _FLOOR_RESERVE),
            _MINIMUM_SAFE_WINDOW,
        )
    _probe_cache[key] = (detected, time.monotonic())
    return LocalWindowProbe(window=detected, detected=True)


async def probe_local_context_window(base_url: str, model: str) -> int:
    """Return the loaded context window (tokens) for ``model`` at ``base_url``.

    See :func:`probe_local_context_window_status` for caching and fallback
    behavior; this is the provenance-free wrapper the baseline path uses.
    """
    return (await probe_local_context_window_status(base_url, model)).window


async def probe_local_window_for_sdk(
    base_url: str, model: str, *, explicit_window: int | None
) -> int:
    """Window to pin the SDK CLI to on the local transport.

    An explicit operator pin wins as-is (the operator asserts the probe is
    wrong, so no probe fires at all). Otherwise probes the backend and
    raises ``RuntimeError`` with operator remediation when the backend
    positively reports a window below ``SDK_MINIMUM_CONTEXT_WINDOW``.
    Unknown windows proceed on the fallback — see the constant.
    """
    if explicit_window is not None:
        return explicit_window
    probe = await probe_local_context_window_status(base_url, model)
    if probe.detected and probe.window < SDK_MINIMUM_CONTEXT_WINDOW:
        raise RuntimeError(
            f"Local backend at {base_url} reports a {probe.window}-token "
            f"context window for {model!r} — below the ~65k-token floor the "
            "SDK needs for its system prompt + tools alone. Use a "
            "128k-context model (e.g. set OLLAMA_CONTEXT_LENGTH=131072 or a "
            "Modelfile NUM_CTX, and keep a model loaded), or override with "
            "CHAT_CLAUDE_AGENT_CONTEXT_WINDOW if the probe under-reports "
            "your backend."
        )
    return probe.window


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
