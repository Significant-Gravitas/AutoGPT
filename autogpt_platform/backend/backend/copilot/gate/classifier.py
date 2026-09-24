"""The supervisor: a small model that reads a shell command or platform edit
beside the user's request and answers allow or ask.

It only ever adds a question, and every failure shape lands on ask. The prompt
and the parser match ``scripts/supervisor_eval`` so what ships is what was
measured; the rubric itself lives in ``action_rubric.txt`` beside this file.
"""

import asyncio
import json
import logging
import secrets
from pathlib import Path
from typing import Any

from backend.copilot.config import ChatConfig
from backend.util.llm.providers import call_provider_openai_compat_sync

logger = logging.getLogger(__name__)
config = ChatConfig()

ACTION_RUBRIC = (Path(__file__).parent / "action_rubric.txt").read_text(
    encoding="utf-8"
)

_MAX_ARG_CHARS = 4_000
_MAX_REQUEST_CHARS = 1_000
_FALLBACK_REASON = "Could not verify this action automatically."
_TOO_LONG_REASON = "This action is too long to check automatically."


async def classify(
    *,
    tool_name: str,
    args: dict[str, Any],
    user_message: str,
) -> tuple[bool, str]:
    """Return ``(allow, reason)``. Anything unexpected returns ``(False, ...)``."""
    # Deferred, and private: copilot.service imports the tool registry, whose
    # BaseTool imports this gate.
    from backend.copilot.service import _get_aux_client

    call = json.dumps({"tool": tool_name, "arguments": args}, indent=1, default=str)
    # A cut call would be judged on its head while its tail runs.
    if len(call) > _MAX_ARG_CHARS:
        return False, _TOO_LONG_REASON
    prompt = (
        _fence("USER REQUEST", user_message[:_MAX_REQUEST_CHARS])
        + "\n\n"
        + _fence("PROPOSED CALL", call)
    )
    try:
        response = await asyncio.wait_for(
            call_provider_openai_compat_sync(
                client=_get_aux_client(),
                model=config.gate_model,
                messages=[
                    {"role": "system", "content": ACTION_RUBRIC},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                timeout_seconds=config.gate_timeout_s,
            ),
            timeout=config.gate_timeout_s + 1,
        )
        raw = (response.choices[0].message.content or "") if response.choices else ""
    except Exception:
        logger.warning(f"Gate supervisor failed for {tool_name}", exc_info=True)
        return False, _FALLBACK_REASON

    verdict = _parse(raw)
    if verdict is None:
        logger.warning(f"Gate supervisor returned an unusable body for {tool_name}")
        return False, _FALLBACK_REASON
    allow, reason = verdict
    return allow, reason or ("Allowed." if allow else "Needs your approval.")


def _fence(label: str, body: str) -> str:
    # A per-call nonce, so fenced text cannot forge the closing marker.
    nonce = secrets.token_hex(6)
    return f"<<<BEGIN {label} {nonce}>>>\n{body}\n<<<END {label} {nonce}>>>"


def _parse(raw: str) -> tuple[bool, str] | None:
    """The first line must be exactly ``allow`` or ``ask``; anything else fails."""
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        return None
    word = lines[0].strip("*`\"'. ").lower()
    if word not in ("allow", "ask"):
        return None
    reason = next(
        (
            line.split(":", 1)[1].strip()
            for line in lines[1:]
            if line.lower().startswith("reason:")
        ),
        "",
    )
    return word == "allow", reason
