"""Otto drives the browser on its desktop: the calls it may make, checked, and
how they run.

The shape is bdrive's (the browser driver in Nick's spark-vm, SECRT-2652), so
that driver can replace this one without the model noticing:

- A call is ``{"actions": [...], "ref_scope": "..."}``: an ordered list of
  actions from a closed vocabulary (``ACTIONS``). An unknown action or
  parameter is refused, never interpreted, and every parameter is bounded.
- Elements are addressed by ``@eN`` refs from the page's accessibility
  snapshot. A ref is valid only with the ``ref_scope`` token of the snapshot
  that gave it, and only the newest snapshot's refs are: a call naming an old
  one is refused before anything runs.
- After a navigation (``goto``, ``back``, ``forward``, ``reload``) nothing in
  the same call may address a ref: the page they came from is gone.
- Each action gets a receipt: ``completed``, ``unknown`` (it may have run;
  look before repeating it) or ``not_started``. The first failure stops the
  call, and every call ends with a fresh snapshot and a new ``ref_scope``.

It runs in the session's own box, on the Chrome the user sees on the desktop
(``start_desktop``): one ``agent-browser`` batch, attached to that Chrome over
its local debugging port. The model has a root shell in the same box, so the
fixed vocabulary is a shape for the model to work in, not a boundary; what
keeps credentials out of the page is the swap proxy, whose placeholders a
later version will let the model type into login forms.
"""

import json
import re
import secrets
from typing import Any, Literal, Optional

from e2b import AsyncSandbox
from pydantic import BaseModel, ConfigDict, Field

from backend.blocks.desktop._api import DISPLAY
from backend.data.redis_client import get_redis_async
from backend.util.e2b_template import AGENT_BROWSER_VERSION
from backend.util.sandbox_login import LoginChainChanged, run_internal
from backend.util.sandbox_output import read_sandbox_file

CDP_PORT = 9222
# agent-browser keeps a daemon per session, attached to one Chrome. A Chrome
# started anew gets a session of its own, recorded here, because the old
# daemon keeps talking to the Chrome that is gone.
SESSION_FILE = "/tmp/otto-browser.session"
PROFILE_DIR = "/home/user/.config/otto-browser"

MAX_ACTIONS = 20
MAX_TEXT = 10_000
MAX_URL = 4_096
MAX_SNAPSHOT_CHARS = 20_000
MAX_TEXT_RESULT_CHARS = 20_000
MAX_SCREENSHOT_BYTES = 10 * 1024 * 1024
# Each action's own limit, unless a ``wait`` sets one; the call's command
# gets the sum plus room for the snapshot that ends it.
DEFAULT_TIMEOUT_MS = 15_000
MAX_WAIT_MS = 60_000
MAX_CALL_SECONDS = 180

_REF_SCOPE_TTL_SECONDS = 3600
_REF_RE = re.compile(r"^@?(e[0-9]{1,6})$")
_KEY_RE = re.compile(r"^[A-Za-z0-9+_ -]{1,64}$")
_NAVIGATIONS = frozenset({"goto", "back", "forward", "reload"})

Status = Literal["completed", "unknown", "not_started"]


class CallRefused(ValueError):
    """The call breaks the protocol; nothing in it ran."""


class _Param(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["ref", "text", "url", "key", "enum", "bool", "ms"]
    required: bool = True
    choices: tuple[str, ...] = ()
    limit: int = MAX_TEXT


def _p(
    kind: Literal["ref", "text", "url", "key", "enum", "bool", "ms"],
    required: bool = True,
    choices: tuple[str, ...] = (),
    limit: int = MAX_TEXT,
) -> _Param:
    return _Param(kind=kind, required=required, choices=choices, limit=limit)


_REF = _p("ref")
_OPTIONAL_REF = _p("ref", required=False)

# The closed vocabulary: each action and the parameters it takes. bdrive's v1
# set, less right-click, which agent-browser has no command for.
ACTIONS: dict[str, dict[str, _Param]] = {
    "open": {},
    "goto": {"url": _p("url", limit=MAX_URL)},
    "back": {},
    "forward": {},
    "reload": {},
    "state": {},
    "click": {"ref": _REF, "kind": _p("enum", False, ("left", "double"))},
    "fill": {"ref": _REF, "text": _p("text")},
    "type": {"ref": _REF, "text": _p("text")},
    "press": {"key": _p("key"), "ref": _OPTIONAL_REF},
    "select": {"ref": _REF, "value": _p("text", limit=1_000)},
    "check": {"ref": _REF},
    "snapshot": {},
    "look": {"full_page": _p("bool", required=False)},
    "get_text": {"ref": _OPTIONAL_REF},
    "wait": {
        "text": _p("text", False, limit=1_000),
        "text_gone": _p("text", False, limit=1_000),
        "time_ms": _p("ms", False, limit=30_000),
        "timeout_ms": _p("ms", False, limit=MAX_WAIT_MS),
    },
    "cookies_clear": {},
}


class Action(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    params: dict[str, Any]

    @property
    def ref(self) -> Optional[str]:
        return self.params.get("ref")


class Receipt(BaseModel):
    action: str
    status: Status
    reason: Optional[str] = None
    result: dict[str, Any] = Field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"action": self.action, "status": self.status}
        if self.reason:
            out["reason"] = self.reason
        if self.result:
            out["result"] = self.result
        return out


class Observation(BaseModel):
    url: str = ""
    title: str = ""
    snapshot: str = ""
    refs: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- validation


def parse_call(
    actions: Any, ref_scope: Any, live_scope: Optional[dict]
) -> list[Action]:
    """The call's actions, checked against the vocabulary and the newest
    snapshot (*live_scope*: ``{"token", "refs"}``, or None before the first).
    Raises ``CallRefused``."""
    if not isinstance(actions, list) or not actions:
        raise CallRefused("actions must be a non-empty list")
    if len(actions) > MAX_ACTIONS:
        raise CallRefused(f"at most {MAX_ACTIONS} actions per call")
    parsed = [_parse_action(i, raw) for i, raw in enumerate(actions)]
    if any(a.name == "snapshot" for a in parsed) and len(parsed) > 1:
        raise CallRefused("snapshot is a call of its own, never mixed with actions")
    if sum(a.name == "look" for a in parsed) > 1:
        raise CallRefused("at most one look per call")
    navigated = False
    for i, action in enumerate(parsed):
        if action.ref is not None:
            if navigated:
                raise CallRefused(
                    f"action {i} ({action.name}) names a ref after a navigation in "
                    "the same call; its page is gone. Take a new snapshot first."
                )
            _check_ref(i, action, ref_scope, live_scope)
        navigated = navigated or action.name in _NAVIGATIONS
    return parsed


def _parse_action(i: int, raw: Any) -> Action:
    if not isinstance(raw, dict) or not isinstance(raw.get("action"), str):
        raise CallRefused(f"action {i} must be an object with an 'action' name")
    name = raw["action"]
    spec = ACTIONS.get(name)
    if spec is None:
        raise CallRefused(f"action {i}: unknown action {name[:40]!r}")
    unknown = sorted(set(raw) - {"action"} - set(spec))
    if unknown:
        shown = ", ".join(repr(k[:40]) for k in unknown[:5])
        raise CallRefused(f"action {i} ({name}): unknown parameters {shown}")
    params: dict[str, Any] = {}
    for key, param in spec.items():
        if key not in raw:
            if param.required:
                raise CallRefused(f"action {i} ({name}): {key} is required")
            continue
        params[key] = _check_param(i, name, key, param, raw[key])
    if name == "wait":
        given = [k for k in ("text", "text_gone", "time_ms") if k in params]
        if len(given) != 1:
            raise CallRefused(
                f"action {i} (wait): exactly one of text, text_gone or time_ms"
            )
        if "time_ms" in params and "timeout_ms" in params:
            raise CallRefused(f"action {i} (wait): timeout_ms is for text waits")
    return Action(name=name, params=params)


def _check_param(i: int, name: str, key: str, param: _Param, value: Any) -> Any:
    where = f"action {i} ({name}): {key}"
    if param.kind == "bool":
        if not isinstance(value, bool):
            raise CallRefused(f"{where} must be true or false")
        return value
    if param.kind == "ms":
        if isinstance(value, bool) or not isinstance(value, int):
            raise CallRefused(f"{where} must be a whole number of milliseconds")
        if not 1 <= value <= param.limit:
            raise CallRefused(f"{where} must be 1 to {param.limit}")
        return value
    if not isinstance(value, str) or not value:
        raise CallRefused(f"{where} must be a non-empty string")
    if len(value) > param.limit:
        raise CallRefused(f"{where} is longer than {param.limit} characters")
    if param.kind == "ref":
        match = _REF_RE.match(value)
        if not match:
            raise CallRefused(f"{where} must be a ref like @e3 from the snapshot")
        return "@" + match.group(1)
    if param.kind == "url":
        if not re.match(r"^https?://", value, re.IGNORECASE):
            raise CallRefused(f"{where} must be an http or https URL")
        return value
    if param.kind == "key":
        if not _KEY_RE.match(value):
            raise CallRefused(f"{where} must be a key name or chord, like Enter")
        return value
    if param.kind == "enum" and value not in param.choices:
        raise CallRefused(f"{where} must be one of {', '.join(param.choices)}")
    return value


def _check_ref(i: int, action: Action, ref_scope: Any, live: Optional[dict]) -> None:
    if not live:
        raise CallRefused(
            f"action {i} ({action.name}) names {action.ref}, but there is no "
            "snapshot yet. Take one first."
        )
    if not isinstance(ref_scope, str) or not secrets.compare_digest(
        ref_scope, live["token"]
    ):
        raise CallRefused(
            f"action {i} ({action.name}): ref_scope is missing or not the newest "
            "snapshot's; only the newest snapshot's refs can be used."
        )
    if (action.ref or "").lstrip("@") not in live["refs"]:
        raise CallRefused(
            f"action {i} ({action.name}): {action.ref} is not in the newest snapshot"
        )


# ---------------------------------------------------------------- translation


def commands_for(action: Action, look_path: str) -> list[list[str]]:
    """The agent-browser commands one action runs as, in order."""
    p, name = action.params, action.name
    if name in ("open", "state"):
        # A bare ``open`` makes agent-browser relaunch the browser it is
        # attached to, which loses the session's tab; ``open`` here only
        # reports where the page is.
        return [["get", "url"], ["get", "title"]]
    if name == "goto":
        return [["open", p["url"]]]
    if name in ("back", "forward", "reload"):
        return [[name]]
    if name == "click":
        return [["dblclick" if p.get("kind") == "double" else "click", p["ref"]]]
    if name in ("fill", "type"):
        return [[name, p["ref"], p["text"]]]
    if name == "press":
        focus = [["focus", p["ref"]]] if "ref" in p else []
        return [*focus, ["press", p["key"]]]
    if name == "select":
        return [["select", p["ref"], p["value"]]]
    if name == "check":
        return [["check", p["ref"]]]
    if name == "snapshot":
        return []
    if name == "look":
        return [["screenshot", *(["--full"] if p.get("full_page") else []), look_path]]
    if name == "get_text":
        return [["get", "text", p.get("ref", "body")]]
    if name == "wait":
        if "time_ms" in p:
            return [["wait", str(p["time_ms"])]]
        # Our own expression around a JSON-quoted string: the model's text
        # is data in it, never code. ``wait --text`` missed text that was
        # plainly on the page in testing; this does not.
        has = f"document.body.innerText.includes({json.dumps(p.get('text') or p['text_gone'])})"
        expression = has if "text" in p else f"!{has}"
        timeout = str(p.get("timeout_ms", DEFAULT_TIMEOUT_MS))
        return [["wait", "--fn", expression, "--timeout", timeout]]
    if name == "cookies_clear":
        return [["cookies", "clear"]]
    raise AssertionError(f"no translation for {name}")


OBSERVE = [["get", "url"], ["get", "title"], ["snapshot", "-i", "-c"]]


def call_seconds(actions: list[Action]) -> int:
    """How long the call's command may take: each action's limit, plus the
    snapshot after."""
    total_ms = 10_000
    for action in actions:
        p = action.params
        total_ms += p.get("time_ms", 0) + p.get("timeout_ms", DEFAULT_TIMEOUT_MS)
    return min(MAX_CALL_SECONDS, total_ms // 1000 + 1)


# ---------------------------------------------------------------- receipts

# agent-browser errors that mean the action never touched the page.
_NOT_STARTED = (
    ("unknown ref", "stale_ref"),
    ("not visible", "not_visible"),
    ("not found", "not_found"),
    ("no element", "not_found"),
    ("disabled", "disabled"),
    ("not editable", "not_editable"),
    ("obscured", "obscured"),
    ("intercepts pointer", "obscured"),
)


def receipts_for(
    actions: list[Action], groups: list[list[list[str]]], entries: list[dict]
) -> list[Receipt]:
    """One receipt per action from the batch's per-command entries, in order.
    The batch stops at the first failing command; nothing after it ran."""
    receipts: list[Receipt] = []
    cursor = 0
    stopped = False
    for action, commands in zip(actions, groups):
        if stopped:
            receipts.append(
                Receipt(
                    action=action.name, status="not_started", reason="earlier_failure"
                )
            )
            continue
        mine = entries[cursor : cursor + len(commands)]
        cursor += len(commands)
        if len(mine) == len(commands) and all(e.get("success") for e in mine):
            receipts.append(
                Receipt(
                    action=action.name, status="completed", result=_result(action, mine)
                )
            )
            continue
        stopped = True
        failed_at = next(
            (i for i, e in enumerate(mine) if not e.get("success")), len(mine)
        )
        error = str(mine[failed_at].get("error")) if failed_at < len(mine) else ""
        receipts.append(_failure(action, error or "no result", ran_any=failed_at > 0))
    return receipts


def _failure(action: Action, error: str, ran_any: bool) -> Receipt:
    """``not_started`` only when the action's first command failed in a way
    that means it never touched the page; anything else may have run."""
    lowered = error.lower()
    if not ran_any:
        for needle, reason in _NOT_STARTED:
            if needle in lowered:
                return Receipt(
                    action=action.name,
                    status="not_started",
                    reason=reason,
                    result={"error": error[:500]},
                )
    reason = "timeout" if "timed out" in lowered else "error"
    return Receipt(
        action=action.name,
        status="unknown",
        reason=reason,
        result={"error": error[:500]},
    )


def _result(action: Action, entries: list[dict]) -> dict[str, Any]:
    results = [e.get("result") or {} for e in entries]
    if action.name in ("open", "state"):
        return {"url": results[0].get("url", ""), "title": results[1].get("title", "")}
    if action.name == "get_text":
        return {"text": str(results[0].get("text", ""))[:MAX_TEXT_RESULT_CHARS]}
    if action.name == "goto":
        return {"url": results[0].get("url", "")}
    return {}


def observation_from(entries: list[dict]) -> Optional[Observation]:
    """The terminal ``get url``, ``get title`` and ``snapshot`` entries."""
    if len(entries) != len(OBSERVE) or not all(e.get("success") for e in entries):
        return None
    url, title, snap = (e.get("result") or {} for e in entries)
    text = str(snap.get("snapshot", ""))
    if len(text) > MAX_SNAPSHOT_CHARS:
        text = text[:MAX_SNAPSHOT_CHARS] + "\n[snapshot cut: the page has more]"
    return Observation(
        url=str(url.get("url", "")),
        title=str(title.get("title", "")),
        snapshot=text,
        refs=sorted((snap.get("refs") or {}).keys()),
    )


def parse_batch(stdout: str) -> list[dict]:
    """The batch's JSON list of ``{command, success, error, result}``; an
    error object (no browser to attach to) reads as one failed entry."""
    text = (stdout or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [{"success": False, "error": text[:500]}]
    if isinstance(data, dict):
        return [data]
    return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []


# ---------------------------------------------------------------- ref scope


def _scope_key(sandbox_id: str) -> str:
    return f"copilot:desktop-browser:ref-scope:{sandbox_id}"


async def live_scope(sandbox_id: str) -> Optional[dict]:
    raw = await (await get_redis_async()).get(_scope_key(sandbox_id))
    if not raw:
        return None
    return json.loads(raw.decode() if isinstance(raw, bytes) else raw)


async def issue_scope(sandbox_id: str, refs: list[str]) -> str:
    """A fresh token for the newest snapshot's refs; every older one is void."""
    token = secrets.token_urlsafe(16)
    await (await get_redis_async()).set(
        _scope_key(sandbox_id),
        json.dumps({"token": token, "refs": refs}),
        ex=_REF_SCOPE_TTL_SECONDS,
    )
    return token


# ---------------------------------------------------------------- the box


class BrowserUnavailable(RuntimeError):
    """The desktop's Chrome could not be started or reached."""


# An idle daemon (its Chrome gone, say) exits after this.
_IDLE_MS = 15 * 60 * 1000


def _ab(session: str) -> str:
    return f"agent-browser --session {session} --cdp {CDP_PORT} --pin-tab"


_CDP_UP = f"curl -sf --max-time 2 http://127.0.0.1:{CDP_PORT}/json/version >/dev/null"
_CHROME = (
    "google-chrome"
    f" --remote-debugging-port={CDP_PORT} --remote-debugging-address=127.0.0.1"
    f" --user-data-dir={PROFILE_DIR}"
    " --no-first-run --no-default-browser-check --disable-quic"
    " --start-maximized about:blank > /tmp/otto-browser.log 2>&1"
)


async def ensure_browser(sandbox: AsyncSandbox) -> str:
    """agent-browser installed and the desktop's Chrome listening on its
    debugging port, started on the screen when it is not (the user may have
    closed it); the agent-browser session attached to that Chrome. A box
    from before the image carried agent-browser gets it now."""
    if not await _ok(
        sandbox, f"agent-browser --version | grep -qF {AGENT_BROWSER_VERSION}"
    ):
        await run_internal(
            sandbox,
            f"npm install -g agent-browser@{AGENT_BROWSER_VERSION} >/dev/null 2>&1",
            user="root",
            timeout=180,
        )
    if await _ok(sandbox, _CDP_UP):
        return await _session(sandbox, fresh=False)
    await run_internal(
        sandbox,
        _CHROME,
        keep_home=True,
        background=True,
        envs={"DISPLAY": DISPLAY},
    )
    if not await _ok(
        sandbox,
        f"for i in $(seq 1 60); do {_CDP_UP} && exit 0; sleep 0.25; done; exit 1",
        timeout=30,
    ):
        raise BrowserUnavailable("Chrome did not start on the desktop")
    session = await _session(sandbox, fresh=True)
    # The new session has no tab yet: bind one now, so the call's first
    # command does not find its pinned tab gone.
    await _ok(sandbox, f"{_ab(session)} tab new about:blank >/dev/null 2>&1")
    return session


async def _session(sandbox: AsyncSandbox, *, fresh: bool) -> str:
    """The session recorded for the running Chrome; a new one when *fresh*
    (that Chrome was just started) or when none is recorded."""
    reset = f"rm -f {SESSION_FILE}; " if fresh else ""
    result = await run_internal(
        sandbox,
        f"{reset}[ -s {SESSION_FILE} ] || "
        f"printf 'otto-%s' \"$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \\n')\" "
        f"> {SESSION_FILE}; cat {SESSION_FILE}",
        keep_home=True,
        timeout=10,
    )
    session = (result.stdout or "").strip()
    if not re.fullmatch(r"otto-[0-9a-f]{12}", session):
        raise BrowserUnavailable("the browser session could not be recorded")
    return session


async def _ok(sandbox: AsyncSandbox, command: str, timeout: int = 20) -> bool:
    try:
        await run_internal(sandbox, command, keep_home=True, timeout=timeout)
        return True
    except LoginChainChanged:
        raise
    except Exception:
        return False


async def run_batch(
    sandbox: AsyncSandbox,
    session: str,
    commands: list[list[str]],
    *,
    bail: bool,
    timeout: int,
) -> list[dict]:
    """One agent-browser batch in the box; its per-command entries. A pinned
    tab that is gone (the user closed it) is bound anew and the batch run
    again: it failed on its first command, so nothing in it ran."""
    if not commands:
        return []
    entries = await _batch_once(sandbox, session, commands, bail=bail, timeout=timeout)
    if entries and entries[0].get("code") == "tab_gone":
        await _ok(sandbox, f"{_ab(session)} tab new about:blank >/dev/null 2>&1")
        entries = await _batch_once(
            sandbox, session, commands, bail=bail, timeout=timeout
        )
    return entries


async def _batch_once(
    sandbox: AsyncSandbox,
    session: str,
    commands: list[list[str]],
    *,
    bail: bool,
    timeout: int,
) -> list[dict]:
    script = f'printf \'%s\' "$OTTO_BATCH" | {_ab(session)} batch {"--bail " if bail else ""}--json'
    try:
        result = await run_internal(
            sandbox,
            script,
            keep_home=True,
            timeout=timeout,
            envs={
                "OTTO_BATCH": json.dumps(commands),
                "AGENT_BROWSER_DEFAULT_TIMEOUT": str(DEFAULT_TIMEOUT_MS),
                "AGENT_BROWSER_IDLE_TIMEOUT_MS": str(_IDLE_MS),
            },
        )
        return parse_batch(result.stdout)
    except LoginChainChanged:
        raise
    except Exception as error:
        # A failed command exits non-zero with its entries on stdout.
        stdout = getattr(error, "stdout", "") or ""
        if stdout.strip():
            return parse_batch(stdout)
        raise


async def read_screenshot(sandbox: AsyncSandbox, path: str) -> Optional[bytes]:
    """The screenshot's bytes, or None when it is over the limit."""
    data = await read_sandbox_file(sandbox, path, MAX_SCREENSHOT_BYTES)
    return None if len(data) > MAX_SCREENSHOT_BYTES else data
