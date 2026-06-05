"""Local LLM routing gate for the copilot.

When the user has a connected local PC shim AND the shim advertises a
working local LLM backend AND the LaunchDarkly flag fires AND the per-
(mode, tier) policy permits it, copilot turns get routed to the shim's
``LOCAL_LLM_COMPLETION`` wire op instead of Anthropic / OpenRouter. The
prompt + response never leave the user's machine.

See ``experimental/local-pc-executor/docs/LOCAL_LLM.md`` for the full
spec — this module implements the activation gate described in
"Activation gate" and "Routing policy" sections.

The router is intentionally a pure decision function (no SDK calls, no
I/O beyond LD evaluation): it returns the model name to route to, or
``None`` to fall back to cloud. The actual streaming integration lives
in ``_LocalLLMProxy.complete()`` on ``LocalPCShim``; ``service.py``
chains the two together via a small SSE-event adapter.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from backend.copilot.config import ChatConfig
from backend.util.feature_flag import Flag, get_feature_flag_value, is_feature_enabled

if TYPE_CHECKING:
    from .local_pc_shim import LocalPCShim

logger = logging.getLogger(__name__)


# Per-tier model preference. Used when the LD flag
# ``LOCAL_LLM_TIER_PREFERENCES`` is missing / malformed. Tiers not in
# this dict (or mapping to []) NEVER route to local — that's the v1
# default for ``thinking`` so long completions stay on the big cloud
# models. Lists are evaluated in order; first match against the shim's
# ``local_llm_models`` advertisement wins.
_DEFAULT_TIER_PREFERENCES: dict[str, list[str]] = {
    "fast": ["llama3.2:3b", "llama3.2:1b", "qwen2.5:3b", "mistral:7b", "phi3:mini"],
    "default": ["llama3.1:8b", "qwen2.5:7b", "mistral:7b"],
    "thinking": [],
}


# Result of a routing decision. Keep it a tuple/dict-free literal so it
# crosses module boundaries cleanly. Returning ``None`` means "fall back
# to cloud"; returning a string means "send LOCAL_LLM_COMPLETION with
# this model".
LocalLLMRoutingDecision = str | None


# Modes the gate understands. Mirrors the copilot's existing
# ``ModelMode`` Literal in model_router.py.
RoutingMode = Literal["fast", "thinking", "default"]


class LocalLLMRouter:
    """When the gate fires, route an LLM completion through the user's shim.

    Pure decision function — no side effects, no I/O beyond LD lookups.
    The chosen model name flows into ``_LocalLLMProxy.complete(...)``;
    ``None`` falls through to the cloud path.
    """

    def __init__(self, config: ChatConfig) -> None:
        self.config = config

    async def should_route(
        self,
        *,
        user_id: str | None,
        mode: str,
        tier: str,
        executor: "LocalPCShim | None",
    ) -> LocalLLMRoutingDecision:
        """Return the shim-side model name to route to, or ``None`` for cloud.

        Gate logic (per docs/LOCAL_LLM.md "Activation gate"):

        1. ``executor`` must be a connected ``LocalPCShim`` (None → cloud).
        2. ``executor.capabilities`` must include ``"local_llm"``.
        3. ``executor.local_llm_models`` must be non-empty.
        4. ``user_id`` must be set AND ``Flag.LOCAL_LLM_ROUTING`` must be
           true for that user.
        5. ``config.local_llm_policy`` must permit this mode:
           - ``"never"``: always cloud.
           - ``"prefer_for_fast"``: local only when ``mode == "fast"``.
           - ``"always"``: any mode.
        6. The first model from the per-tier preference list that's also
           in ``executor.local_llm_models`` is returned. If none match,
           we fall back to cloud — we don't pick a random model just
           because routing was theoretically enabled.

        All failures log at DEBUG so production observability still
        surfaces "why didn't local fire" without spamming.
        """
        # Gate 1: executor must exist
        if executor is None:
            logger.debug("[LocalLLM] No executor — cloud route")
            return None

        # Gate 2: capability advertised
        caps = getattr(executor, "capabilities", None) or []
        if "local_llm" not in caps:
            logger.debug("[LocalLLM] Shim missing local_llm capability — cloud route")
            return None

        # Gate 3: at least one model loaded
        advertised = list(getattr(executor, "local_llm_models", None) or [])
        if not advertised:
            logger.debug("[LocalLLM] Shim has empty local_llm_models — cloud route")
            return None

        # Gate 4: LD flag per user
        if not user_id:
            logger.debug("[LocalLLM] No user_id — cloud route")
            return None
        if not await is_feature_enabled(Flag.LOCAL_LLM_ROUTING, user_id, default=False):
            logger.debug("[LocalLLM] LD flag off for user — cloud route")
            return None

        # Gate 5: policy
        policy = self.config.local_llm_policy
        if policy == "never":
            logger.debug("[LocalLLM] Policy=never — cloud route")
            return None
        if policy == "prefer_for_fast" and mode != "fast":
            logger.debug(
                "[LocalLLM] Policy=prefer_for_fast but mode=%s — cloud route", mode
            )
            return None

        # Gate 6: tier-matched model. Tiers we don't recognise fall back to
        # "default"; tiers explicitly mapped to [] (e.g. thinking by default)
        # are an intentional opt-out and we DON'T sub in the default list.
        preferences = await self._tier_preferences(user_id)
        if tier in preferences:
            tier_list = preferences[tier]
        else:
            tier_list = preferences.get("default") or []
        if not tier_list:
            logger.debug(
                "[LocalLLM] No tier preferences for tier=%s — cloud route", tier
            )
            return None
        for candidate in tier_list:
            if candidate in advertised:
                logger.debug(
                    "[LocalLLM] Routing locally — mode=%s tier=%s model=%s",
                    mode,
                    tier,
                    candidate,
                )
                return candidate
        logger.debug(
            "[LocalLLM] No tier-%s model overlap with advertised=%s — cloud route",
            tier,
            advertised,
        )
        return None

    async def _tier_preferences(self, user_id: str) -> dict[str, list[str]]:
        """Fetch the per-tier preference dict from LD, validate, fall back
        to defaults on any shape problem."""
        try:
            payload = await get_feature_flag_value(
                Flag.LOCAL_LLM_TIER_PREFERENCES.value,
                user_id,
                default=None,
            )
        except Exception as exc:
            logger.debug("[LocalLLM] tier-prefs LD lookup failed: %s", exc)
            return _DEFAULT_TIER_PREFERENCES
        if not isinstance(payload, dict):
            return _DEFAULT_TIER_PREFERENCES
        cleaned: dict[str, list[str]] = {}
        for tier, value in payload.items():
            if not isinstance(tier, str):
                continue
            if not isinstance(value, list):
                continue
            models = [m for m in value if isinstance(m, str)]
            cleaned[tier] = models
        # Fill in unmentioned tiers from defaults so callers can rely on
        # the dict having a known shape (incl. ``thinking: []``).
        for tier, default_list in _DEFAULT_TIER_PREFERENCES.items():
            cleaned.setdefault(tier, default_list)
        return cleaned


__all__ = [
    "LocalLLMRouter",
    "LocalLLMRoutingDecision",
    "RoutingMode",
]
