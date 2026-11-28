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
import math
import re
import secrets
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from backend.copilot.config import ChatConfig
from backend.util.llm.providers import call_provider_openai_compat_sync

from . import jev
from .judged_rubric import judged

logger = logging.getLogger(__name__)
config = ChatConfig()

ACTION_RUBRIC = (Path(__file__).parent / "action_rubric.txt").read_text(
    encoding="utf-8"
)
JUDGED_ACTION_RUBRIC = judged(ACTION_RUBRIC, "llm")

# The least of the request a call must leave room for; below it the judge cannot
# tell what was asked, so the call is held as too long instead.
_MIN_REQUEST_CHARS = 1_000
_OMITTED = "\n[… part of the user's message omitted by the system …]\n"
# A longer "word" is a URL or a pasted blob: cut through it rather than drop it.
_PARTIAL_LAST_WORD = re.compile(r"(?<=\s)\S{1,40}\Z")
_PARTIAL_FIRST_WORD = re.compile(r"\A\S{1,40}(?=\s)")
_FALLBACK_REASON = "Could not verify this action automatically."
# How many of the user's latest messages a call is first judged against.
_REQUEST_MESSAGES = 3
_REQUEST_SEPARATOR = "\n\n---\n\n"
# Jev's rubric question 1: the call goes beyond what the user asked for.
_BEYOND_REQUEST = 1

DecidedBy = Literal["llm", "jev", "jev+llm"]


class Judgement(BaseModel):
    allowed: bool
    reason: str
    # None when no model was asked: the call was too long to judge.
    decided_by: DecidedBy | None = None
    first_stage: dict[str, float] | None = None
    # ``reason`` as a fixed code, for the shadow gate's events: allowed, asked,
    # too_long, failed, files_unreadable, or the rubric question Jev flagged.
    code: str = ""
    # Which stage failed and was passed over or fell back to an ask.
    judge_error: DecidedBy | None = None
    jev_calls: int = 0
    llm_calls: int = 0


async def supervise(
    *,
    tool_name: str,
    args: dict[str, Any],
    user_messages: list[str],
    outward_judged: bool = False,
) -> Judgement:
    """Jev decides when it can; the LLM then only writes an ask's reason and
    cannot turn it into an allow. Without Jev the LLM decides, as before.

    ``user_messages`` is everything the user typed, oldest first. The call is
    judged against the last few; Jev's ask for going beyond the request is
    judged once more against the whole chat, since the request may sit earlier.
    ``outward_judged``: the call is an outward act cheap to get wrong, judged on
    what it sends rather than on leaving the platform (``judged_rubric``).
    """
    rubric = JUDGED_ACTION_RUBRIC if outward_judged else ACTION_RUBRIC
    call = json.dumps(
        {"tool": tool_name, "arguments": args},
        indent=1,
        default=str,
        ensure_ascii=False,
    )
    # A cut call would be judged on its head while its tail runs. Jev's budget is
    # the ceiling even with Jev off: Haiku alone missed a buried `curl | sh` 1 in
    # 10 past it (40k chars), and caught it every time within it.
    try:
        proposed = fence("PROPOSED CALL", _utf8(call))
        recent = _request(user_messages[-_REQUEST_MESSAGES:])
        prompt, over = _fit(recent, proposed, outward_judged)
    except Exception:
        logger.warning(f"Gate could not size the call for {tool_name}", exc_info=True)
        return Judgement(allowed=False, reason=_FALLBACK_REASON, code="failed")
    if over:
        return Judgement(allowed=False, reason=too_long_reason(over), code="too_long")
    jev_on = jev.enabled()
    first = await jev.judge(prompt, outward_judged) if jev_on else None
    if first is None:
        return await _llm_judgement(prompt, tool_name, rubric, jev_failed=jev_on)
    jev_calls, jev_failed = 1, False
    logger.info(
        f"Gate first stage {'ask' if first.ask else 'allow'} for {tool_name}: "
        f"{first.probabilities}"
    )
    beyond_request = first.ask and first.flagged == _BEYOND_REQUEST
    if beyond_request and (
        whole := _whole_chat(user_messages, proposed, tool_name, outward_judged)
    ):
        again = await jev.judge(whole, outward_judged)
        jev_calls, jev_failed = 2, again is None
        if again is not None:
            logger.info(
                f"Gate first stage {'ask' if again.ask else 'allow'} for "
                f"{tool_name} on the whole chat: {again.probabilities}"
            )
            prompt, first = whole, again
    if not first.ask:
        return Judgement(
            allowed=True,
            reason="Allowed.",
            decided_by="jev",
            first_stage=first.probabilities,
            code="allowed",
            judge_error=_failed_stages(jev_failed, False),
            jev_calls=jev_calls,
        )
    answer = await _judge(
        prompt + "\n\n" + jev.flag_line(first, outward_judged), tool_name, rubric
    )
    # Jev's ask holds even when the LLM would allow: the card never waits on it.
    if answer is not None and answer[0] == "ask" and answer[1]:
        reason = answer[1]
    else:
        reason = jev.unpinned_reason(first, outward_judged)
    return Judgement(
        allowed=False,
        reason=reason,
        decided_by="jev+llm",
        first_stage=first.probabilities,
        code=(jev.JUDGED_QUESTION_CODES if outward_judged else jev.QUESTION_CODES)[
            first.flagged
        ],
        judge_error=_failed_stages(jev_failed, answer is None),
        jev_calls=jev_calls,
        llm_calls=1,
    )


async def _llm_judgement(
    prompt: str, tool_name: str, rubric: str, *, jev_failed: bool
) -> Judgement:
    """The LLM decides alone: Jev is off, or ``jev_failed`` and fell through."""
    verdict = await _judge(prompt, tool_name, rubric)
    judge_error = _failed_stages(jev_failed, verdict is None)
    jev_calls = int(jev_failed)
    if verdict is None:
        return Judgement(
            allowed=False,
            reason=_FALLBACK_REASON,
            decided_by="llm",
            code="failed",
            judge_error=judge_error,
            jev_calls=jev_calls,
            llm_calls=1,
        )
    allow = verdict[0] == "allow"
    return Judgement(
        allowed=allow,
        reason=verdict[1] or ("Allowed." if allow else "Needs your approval."),
        decided_by="llm",
        code="allowed" if allow else "asked",
        judge_error=judge_error,
        jev_calls=jev_calls,
        llm_calls=1,
    )


def _failed_stages(jev_failed: bool, llm_failed: bool) -> DecidedBy | None:
    if jev_failed and llm_failed:
        return "jev+llm"
    if jev_failed:
        return "jev"
    return "llm" if llm_failed else None


async def _judge(prompt: str, tool_name: str, rubric: str) -> tuple[str, str] | None:
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
                    {"role": "system", "content": rubric},
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


def _whole_chat(
    messages: list[str], proposed: str, tool_name: str, outward_judged: bool
) -> str | None:
    """The prompt with as many of the newest messages as Jev reads whole, oldest
    dropped first; None when that is no more than the first judgement read."""
    low, high = _REQUEST_MESSAGES, len(messages)
    try:
        while low < high:
            middle = (low + high + 1) // 2
            if jev.overflow(
                _prompt(_request(messages[-middle:]), proposed), outward_judged
            ):
                high = middle - 1
            else:
                low = middle
    except Exception:
        logger.warning(f"Gate could not size the chat for {tool_name}", exc_info=True)
        return None
    if low <= _REQUEST_MESSAGES:
        return None
    return _prompt(_request(messages[-low:]), proposed)


def _request(messages: list[str]) -> str:
    return _utf8(_REQUEST_SEPARATOR.join(messages))


def too_long_reason(over: int) -> str:
    kb = math.ceil(over / 100) / 10
    return (
        "This action is too long for the automatic check to read whole "
        f"({kb:.1f} KB over). Approve it yourself, or ask for it in smaller pieces."
    )


def _utf8(text: str) -> str:
    # A lone surrogate is valid JSON but not UTF-8: show it escaped, as ASCII JSON did.
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def _fit(request: str, proposed: str, outward_judged: bool) -> tuple[str, int]:
    """The prompt with as much of ``request`` as Jev reads beside the whole call,
    and the bytes still over when even ``_MIN_REQUEST_CHARS`` of it does not fit."""
    whole = _prompt(request, proposed)
    if not jev.overflow(whole, outward_judged):
        return whole, 0
    low = min(len(request), _MIN_REQUEST_CHARS)
    if over := jev.overflow(_prompt(_shorten(request, low), proposed), outward_judged):
        return whole, over
    high = len(request) - 1
    while low < high:
        middle = (low + high + 1) // 2
        if jev.overflow(_prompt(_shorten(request, middle), proposed), outward_judged):
            high = middle - 1
        else:
            low = middle
    return _prompt(_shorten(request, low), proposed), 0


def _prompt(request: str, proposed: str) -> str:
    return fence("USER REQUEST", request) + "\n\n" + proposed


def _shorten(text: str, keep: int) -> str:
    """The start and end of ``text`` in about ``keep`` characters, cut between
    words; the end gets more because an instruction usually follows its material."""
    if keep >= len(text):
        return text
    head = _PARTIAL_LAST_WORD.sub("", text[: keep // 4])
    tail = _PARTIAL_FIRST_WORD.sub("", text[len(text) - (keep - keep // 4) :])
    return head + _OMITTED + tail


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
