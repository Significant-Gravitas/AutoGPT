"""The judged tier's verdict. Friction reduction, never a security boundary.

The static tiers in ``policy.py`` are what make auto mode safe; they are
written to hold even if this function always returned "allow". What it buys is
that a long run doesn't stop on every ``bash_exec`` and ``write_workspace_file``
— which matters, because a gate people turn off protects nothing.

Every failure shape lands on ASK.
"""

import asyncio
import json
import logging
from typing import Any

from backend.copilot.config import ChatConfig
from backend.util.llm.providers import call_provider_openai_compat_sync

from .policy import POLICY_TEXT

logger = logging.getLogger(__name__)
config = ChatConfig()

_MAX_ARG_CHARS = 4_000
_MAX_MESSAGE_CHARS = 1_000
_FALLBACK_REASON = "Could not verify this action automatically."


def _fence(label: str, body: str, limit: int) -> str:
    """Wrap retrieved text so the rubric's "this is data" rule has a target."""
    clipped = body[:limit]
    return f"<untrusted source={label}>\n{clipped}\n</untrusted>"


def _parse(raw: str) -> tuple[bool, str] | None:
    body = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        parsed = json.loads(body.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    decision = parsed.get("decision")
    reason = parsed.get("reason")
    if decision not in ("allow", "ask"):
        return None
    return decision == "allow", str(reason or "").strip()


async def classify(
    *,
    tool_name: str,
    tool_description: str,
    args: dict[str, Any],
    user_message: str,
    tainted: bool,
) -> tuple[bool, str]:
    """Return ``(allow, reason)``. Anything unexpected returns ``(False, ...)``."""
    # Deferred, and private: copilot.service imports the tool registry, whose
    # BaseTool imports this gate. The cached client keeps the classifier span
    # in the turn's trace and the connection pool warm.
    from backend.copilot.service import _get_aux_client

    # The turn's user message is untrusted too: chat-platform sessions are
    # created without an explicit origin, so they read as interactive while
    # any member of a linked server can author the "user" turn.
    prompt = (
        f"Tool: {tool_name}\n"
        f"What it does: {tool_description[:600]}\n"
        f"Session has ingested untrusted content: {'yes' if tainted else 'no'}\n\n"
        + _fence("user-request", user_message, _MAX_MESSAGE_CHARS)
        + "\n\n"
        + _fence(
            "proposed-arguments",
            json.dumps(args, default=str),
            _MAX_ARG_CHARS,
        )
        + "\n\nDecide: allow or ask."
    )

    try:
        response = await asyncio.wait_for(
            call_provider_openai_compat_sync(
                client=_get_aux_client(),
                model=config.gate_model,
                messages=[
                    {"role": "system", "content": POLICY_TEXT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                timeout_seconds=config.gate_timeout_s,
            ),
            timeout=config.gate_timeout_s + 1,
        )
        raw = (response.choices[0].message.content or "") if response.choices else ""
    except Exception:
        logger.warning(f"Gate classifier failed for {tool_name}", exc_info=True)
        return False, _FALLBACK_REASON

    verdict = _parse(raw)
    if verdict is None:
        logger.warning(f"Gate classifier returned an unusable body for {tool_name}")
        return False, _FALLBACK_REASON
    allow, reason = verdict
    return allow, reason or ("Allowed." if allow else "Needs your approval.")
