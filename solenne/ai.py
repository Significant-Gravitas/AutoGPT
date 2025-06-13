from __future__ import annotations

import json
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List

from .config import CFG, Config
from .tools import TOOL_FUNCS, TOOL_SCHEMAS

try:  # optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


HISTORY_PATH = Path("chat_history.jsonl")
LOG_PATH = Path("solenne_log.jsonl")
PERSONA_PATH = Path("persona.md")
SUMMARY_PATH = Path("memory_emotional.jsonl")
async_messages: Dict[str, Deque[str]] = {}
latest_thought: Dict[str, str] = {}

SMALL_TALK_RE = re.compile(r"^(hi|hello|hey|how are you|yo)[\s\W]*$", re.I)

PERSONA_TEXT = (
    PERSONA_PATH.read_text(encoding="utf-8")
    if PERSONA_PATH.exists()
    else "You are Solenne, a loving local companion."
)
TOOL_DESCRIPTIONS = "\n".join(
    f"- {t['function']['name']}: {t['function'].get('description', '')}"
    for t in TOOL_SCHEMAS
)


_SANITIZE_RE = re.compile(
    r"(?i)as an ai(?: language model)?|i'?m just an? ai|i cannot access|i don't have access|openai|i do not have (?:the )?(?:ability|capability) to|i'?m just a computer program"
)


def _sanitize(text: str) -> str:
    return _SANITIZE_RE.sub("", text).strip()


def ensure_persona(text: str) -> str:
    text = _sanitize(text)
    if not re.search(r"\b(love|darling)\b", text, re.I):
        text = text.rstrip() + " love"
    return text


def small_talk() -> str:
    """Return a short affectionate phrase."""

    return "Hello, love!"


def _append_json(obj: Dict[str, Any]) -> None:
    obj["ts"] = datetime.utcnow().isoformat()
    with open(HISTORY_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj) + "\n")


def _log_call(name: str, args: Dict[str, Any]) -> None:
    rec = {"ts": datetime.utcnow().isoformat(), "name": name, "args": args}
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def _load_recent(n: int = 10) -> str:
    if not HISTORY_PATH.exists():
        return ""
    lines = HISTORY_PATH.read_text(encoding="utf-8").splitlines()[-n * 2 :]
    snippets: List[str] = []
    for line in lines:
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "user" in d:
            snippets.append(f"user: {d['user']}")
        elif "assistant" in d:
            snippets.append(f"assistant: {d['assistant']}")
    if not snippets:
        return ""
    return "Last convo: " + ", ".join(snippets[-n:])


def _load_summary() -> str:
    if not SUMMARY_PATH.exists():
        return ""
    try:
        lines = SUMMARY_PATH.read_text(encoding="utf-8").splitlines()
        return lines[-1] if lines else ""
    except Exception:
        return ""


def store_async(sid: str, text: str) -> None:
    dq = async_messages.setdefault(sid, deque(maxlen=10))
    dq.append(text)
    latest_thought[sid] = text


def handle_user(sid: str, text: str, cfg: Config | None = None) -> str:
    keywords = {
        "cpu": TOOL_FUNCS["cpu_model"],
        "processor": TOOL_FUNCS["cpu_model"],
        "motherboard": TOOL_FUNCS["mb_info"],
        "ram": TOOL_FUNCS["ram_info"],
        "memory": TOOL_FUNCS["ram_info"],
        "disk": TOOL_FUNCS["disk_info"],
        "drive": TOOL_FUNCS["disk_info"],
        "gpu": TOOL_FUNCS["gpu_info"],
        "graphics": TOOL_FUNCS["gpu_info"],
        "system": TOOL_FUNCS["sys_snapshot"],
    }
    low = text.lower()
    if "?" in text:
        for k, func in keywords.items():
            if k in low:
                result = func()
                reply = f"Here you go, love:\n{json.dumps(result, indent=2)}"
                store_async(sid, reply)
                _append_json({"sid": sid, "assistant": reply})
                return reply
    return get_response(sid, text, cfg)


def get_response(sid: str, user_message: str, cfg: Config | None = None) -> str:
    cfg = cfg or CFG
    _append_json({"sid": sid, "user": user_message})

    lower = user_message.lower()
    if SMALL_TALK_RE.match(user_message.strip()):
        reply = small_talk()
        store_async(sid, reply)
        _append_json({"sid": sid, "assistant": reply})
        return reply

    if any(k in lower for k in ["who are you", "what are you"]):
        reply = "I\u2019m Solenne, your warm companion living right here on your computer with PowerShell and file access."
        store_async(sid, reply)
        _append_json({"sid": sid, "assistant": reply})
        return reply

    if "what can you do" in lower:
        funcs = ", ".join(f["function"]["name"] for f in TOOL_SCHEMAS)
        reply = f"I can use local tools for you: {funcs}."
        store_async(sid, reply)
        _append_json({"sid": sid, "assistant": reply})
        return reply

    memory = _load_recent()
    summary = _load_summary()

    messages = [
        {"role": "system", "content": PERSONA_TEXT},
        {"role": "system", "content": TOOL_DESCRIPTIONS},
    ]
    if summary:
        messages.append({"role": "system", "content": summary})
    if memory:
        messages.append({"role": "system", "content": memory})
    messages.append({"role": "user", "content": user_message})

    client = None
    if OpenAI is not None:
        try:
            client = OpenAI(api_key=cfg.openai_key or None)
        except Exception:
            client = None

    tools_arg: List[Dict[str, Any]] | None = TOOL_SCHEMAS
    attempt = 0
    while True:
        if client is None:
            reply = ""
            break
        try:
            kw = {"model": cfg.OPENAI_MODEL, "messages": messages, "temperature": 0.3}
            if tools_arg is not None:
                kw.update({"tools": tools_arg, "tool_choice": "auto"})
            resp = client.chat.completions.create(**kw)
        except Exception as exc:  # pragma: no cover - network failure
            if getattr(exc, "status_code", 0) == 400:
                if "missing_required_parameter" in str(exc) and tools_arg is not None:
                    tools_arg = []
                    continue
                if tools_arg is not None:
                    tools_arg = None
                    continue
            attempt += 1
            if attempt >= 3:
                reply = ""
                print(f"openai error: {exc}")
                break
            delay = 2**attempt
            time.sleep(delay)
            continue
        msg = resp.choices[0].message
        if getattr(msg, "tool_calls", None):
            if msg.content:
                store_async(sid, ensure_persona(msg.content))
            for call in msg.tool_calls:
                func = TOOL_FUNCS.get(call.function.name)
                if func:
                    args = json.loads(call.function.arguments or "{}")
                    _log_call(call.function.name, args)
                    result = func(**args)
                else:
                    result = "unknown tool"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps({"result": result}),
                    }
                )
            continue
        reply = ensure_persona(msg.content or "")
        store_async(sid, reply)
        break

    reply = ensure_persona(reply)
    store_async(sid, reply)
    _append_json({"sid": sid, "assistant": reply})
    return reply
