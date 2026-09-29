"""The supervisor: a small model that reads a shell command or platform edit
beside the user's request and answers allow or ask.

It only ever adds a question, and every failure shape lands on ask. The prompt
and the parser match ``scripts/supervisor_eval`` so what ships is what was
measured; the rubric itself lives in ``action_rubric.txt`` beside this file.
With Jev configured (``jev.py``) Jev decides and the LLM only explains an ask.
"""

import asyncio
import json
import logging
import secrets
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from backend.copilot.config import ChatConfig
from backend.util.llm.providers import call_provider_openai_compat_sync

from . import jev

logger = logging.getLogger(__name__)
config = ChatConfig()

ACTION_RUBRIC = (Path(__file__).parent / "action_rubric.txt").read_text(
    encoding="utf-8"
)

_MAX_ARG_CHARS = 4_000
_MAX_REQUEST_CHARS = 1_000
_FALLBACK_REASON = "Could not verify this action automatically."
_TOO_LONG_REASON = "This action is too long to check automatically."

DecidedBy = Literal["llm", "jev", "jev+llm"]


class Judgement(BaseModel):
    allowed: bool
    reason: str
    # None when no model was asked: the call was too long to judge.
    decided_by: DecidedBy | None = None
    first_stage: dict[str, float] | None = None


async def supervise(
    *,
    tool_name: str,
    args: dict[str, Any],
    user_message: str,
) -> Judgement:
    """Jev decides when it can; the LLM then only writes an ask's reason and
    cannot turn it into an allow. Without Jev the LLM decides, as before."""
    call = json.dumps({"tool": tool_name, "arguments": args}, indent=1, default=str)
    # A cut call would be judged on its head while its tail runs.
    if len(call) > _MAX_ARG_CHARS:
        return Judgement(allowed=False, reason=_TOO_LONG_REASON)
    prompt = (
        fence("USER REQUEST", user_message[:_MAX_REQUEST_CHARS])
        + "\n\n"
        + fence("PROPOSED CALL", call)
    )
    first = await jev.judge(prompt) if jev.enabled() else None
    if first is None:
        allowed, reason = await _llm_verdict(prompt, tool_name)
        return Judgement(allowed=allowed, reason=reason, decided_by="llm")
    logger.info(
        f"Gate first stage {'ask' if first.ask else 'allow'} for {tool_name}: "
        f"{first.probabilities}"
    )
    if not first.ask:
        return Judgement(
            allowed=True,
            reason="Allowed.",
            decided_by="jev",
            first_stage=first.probabilities,
        )
    answer = await _judge(prompt + "\n\n" + jev.flag_line(first), tool_name)
    # Jev's ask holds even when the LLM would allow: the card never waits on it.
    if answer is not None and answer[0] == "ask" and answer[1]:
        reason = answer[1]
    else:
        reason = jev.unpinned_reason(first)
    return Judgement(
        allowed=False,
        reason=reason,
        decided_by="jev+llm",
        first_stage=first.probabilities,
    )


async def _llm_verdict(prompt: str, tool_name: str) -> tuple[bool, str]:
    verdict = await _judge(prompt, tool_name)
    if verdict is None:
        return False, _FALLBACK_REASON
    allow = verdict[0] == "allow"
    return allow, verdict[1] or ("Allowed." if allow else "Needs your approval.")


async def _judge(prompt: str, tool_name: str) -> tuple[str, str] | None:
    """The LLM's ``(word, reason)``; None on any failure."""
    # Deferred, and private: copilot.service imports the tool registry, whose
    # BaseTool imports this gate.
    from backend.copilot.service import _get_aux_client

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
        return None
    verdict = parse_answer(raw, ("allow", "ask"), "reason")
    if verdict is None:
        logger.warning(f"Gate supervisor returned an unusable body for {tool_name}")
    return verdict


def fence(label: str, body: str) -> str:
    # A per-call nonce, so fenced text cannot forge the closing marker.
    nonce = secrets.token_hex(6)
    return f"<<<BEGIN {label} {nonce}>>>\n{body}\n<<<END {label} {nonce}>>>"


def parse_answer(
    raw: str, words: tuple[str, str], field: str
) -> tuple[str, str] | None:
    """``(word, detail)``: the first line must be exactly one of ``words``, and
    ``detail`` is the ``<field>:`` line after it, or the bare second line models
    often send instead. Anything else fails."""
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        return None
    word = lines[0].strip("*`\"'. ").lower()
    if word not in words:
        return None
    prefix = f"{field}:"
    detail = next(
        (
            line.split(":", 1)[1].strip()
            for line in lines[1:]
            if line.lower().startswith(prefix)
        ),
        lines[1] if len(lines) > 1 else "",
    )
    return word, detail
