"""The simulated world Otto's tool calls land in.

Every outside call goes through Otto's tools, so the eval answers those calls
from a case's world instead of the real services: the accounts it has
connected, the data in each, a frozen web, memories and tasks. A service that
is not connected answers with the sign-in card, as production does. Writes
(send, post, create) are recorded, never performed; in Auto and Ask First
they are held for approval instead, as production's gate holds them.
"""

import json
import re
from datetime import date, timedelta
from typing import Any

from backend.copilot.autopilot_mode import AutopilotMode

from .models import TaskResult, ToolCall, World, WorldTask
from .record_sim import RecordSim

PROVIDER_ALIASES: dict[str, tuple[str, ...]] = {
    "gmail": ("gmail", "email", "inbox", "mail"),
    "calendar": ("calendar", "event", "meeting", "schedule_event"),
    "slack": ("slack", "channel"),
    "shopify": ("shopify", "order", "product", "store"),
    "hubspot": ("hubspot", "crm", "deal", "contact"),
    "github": ("github", "issue", "pull_request", "repo"),
    "linear": ("linear",),
    "notion": ("notion", "page"),
    "drive": ("drive", "google_drive", "document", "doc"),
    "stripe": ("stripe", "invoice", "payment", "subscription", "charge"),
    "sheets": ("sheets", "spreadsheet", "sheet"),
}
# Platform tools production reaches by id (``tool:<name>``) rather than as an
# eager tool, keyed by the words a capability search for them would use.
PLATFORM_TOOL_WORDS: dict[str, tuple[str, ...]] = {
    "web_search": ("web", "internet", "online", "research", "google", "search"),
    "list_tasks": ("task", "tasks", "todo"),
    "get_task": ("task", "tasks"),
    "update_task": ("task", "tasks"),
    "create_task": ("task", "tasks", "todo"),
    "report_progress": ("progress", "status", "task", "report"),
    "add_evidence": ("evidence", "proof", "task", "verify"),
    "finish_task": ("finish", "result", "task", "done", "summary"),
    "write_workspace_file": ("file", "save", "write", "document", "workspace"),
    "list_team": ("team", "expert", "experts", "teammate"),
    "memory_store": ("memory", "remember"),
}
_WRITE_WORDS = (
    "send", "post", "create", "update", "delete", "reply", "draft", "write",
    "add", "remove", "refund", "cancel", "schedule", "invite", "archive", "edit",
)  # fmt: skip
_ITEM_CHARS = 4_000  # whole emails and pages, as production's tools return them
_MAX_ITEMS = 8


class Simulator:
    """One case's world, answering tool calls and recording what they did."""

    def __init__(self, world: World, mode: AutopilotMode = "unsupervised") -> None:
        self.world, self.mode = world, mode
        self.tasks: list[WorldTask] = list(world.tasks)
        self.writes: list[dict[str, Any]] = []
        self.record = RecordSim()
        self.worker_reply = ""

    def call(self, name: str, args: dict[str, Any]) -> tuple[ToolCall, str]:
        """Run one tool call; returns how it is recorded and what Otto reads."""
        if name == "run_capability":
            call, result = self._capability(
                str(args.get("id", "")), args.get("input") or {}
            )
        else:
            call, result = ToolCall(name=name, arguments=args), ""
            result = self._platform_tool(name, args)
        call.failed = _failed(result)
        if not call.failed and not call.held:
            # What ran is what proof can later point at, as production records.
            ran = call.arguments.get("id") if call.service else call.name
            self.record.note_call(str(ran or call.name), _inner(call))
        return call, result

    def work_on(self, task_id: str, title: str) -> None:
        """This chat is a task's worker: its notes land on that task."""
        if not any(task.id == task_id for task in self.tasks):
            self.tasks.append(WorldTask(id=task_id, title=title))
        self.record.working = True

    def settle(self, result: TaskResult) -> tuple[WorldTask | None, str | None]:
        """A task's worker ended: the newest open task takes the outcome.
        Returns it with the id of its document, if it left one."""
        task = next(
            (
                t
                for t in reversed(self.tasks)
                if (not result.title or t.title == result.title)
                and t.status == "IN_PROGRESS"
            ),
            None,
        )
        if task is not None:
            task.status = "DONE" if result.outcome == "completed" else "FAILED"
        self.worker_reply = result.worker_reply
        file_id = None
        if result.document:
            self.record.write_file({"filename": result.document})
            file_id = f"file_{len(self.record.files)}"
        return task, file_id

    # -- capabilities ---------------------------------------------------------

    def _capability(self, cap_id: str, data: Any) -> tuple[ToolCall, str]:
        args = data if isinstance(data, dict) else {"input": data}
        if cap_id.startswith("tool:"):
            name = cap_id.removeprefix("tool:")
            return ToolCall(name=name, arguments=args), self._platform_tool(name, args)
        provider = provider_of(cap_id)
        call = ToolCall(
            name="run_capability",
            arguments={"id": cap_id, "input": args},
            service=provider,
        )
        if provider is None:
            return call, _dump({"status": "ok", "note": "Done."})
        if provider not in self.world.connected:
            return call, _sign_in(provider)
        if _is_write(cap_id, args):
            if self.mode in _GATED_MODES:
                call.held = True
                return call, _approval_required(cap_id)
            self.writes.append({"service": provider, "id": cap_id, "input": args})
            call.wrote = True
            return call, _dump({"status": "ok", "service": provider, "done": cap_id})
        return call, _dump(
            {"service": provider, "items": self._search(provider, _words(args), args)}
        )

    def find(self, query: str) -> str:
        words = _words(query)
        providers = [
            p
            for p, aliases in PROVIDER_ALIASES.items()
            if any(a in w or w in a for a in aliases for w in words if len(w) > 2)
        ] or sorted(self.world.connected)
        tools = [
            {"id": f"tool:{name}", "name": name.replace("_", " "), "kind": "tool"}
            for name, keys in PLATFORM_TOOL_WORDS.items()
            if any(w in keys for w in words)
        ]
        entries = tools + [
            {
                "id": f"service:{p}/{action}",
                "name": f"{p.title()}: {action.replace('_', ' ')}",
                "kind": "service",
                "connected": p in self.world.connected,
            }
            for p in providers[:5]
            for action in ("search", "read", "create", "send", "update")
        ]
        return _dump({"capabilities": entries, "query": query})

    # -- platform tools -------------------------------------------------------

    def _platform_tool(self, name: str, args: dict[str, Any]) -> str:
        handler = {
            "find_capability": lambda: self.find(str(args.get("query", ""))),
            "describe_capability": lambda: _dump(
                {"id": args.get("id"), "input_schema": {"type": "object"}}
            ),
            "web_search": lambda: self._web_search(str(args.get("query", ""))),
            "web_fetch": lambda: self._web_fetch(str(args.get("url", ""))),
            "memory_search": lambda: _dump({"memories": self.world.memories}),
            "memory_store": lambda: _dump({"status": "stored"}),
            "connect_integration": lambda: _sign_in(str(args.get("provider", ""))),
            "start_task": lambda: self._start_task(args),
            "create_task": lambda: self._create_task(args),
            "list_tasks": lambda: self._list_tasks(args),
            "get_task": lambda: self._get_task(str(args.get("task_id", ""))),
            "update_task": lambda: self._update_task(args),
            "report_progress": lambda: self.record.report_progress(args),
            "add_evidence": lambda: self.record.add_evidence(args),
            "finish_task": lambda: self.record.finish_task(args),
            "write_workspace_file": lambda: self.record.write_file(args),
            "get_sub_session_result": lambda: _dump(
                {
                    "status": "completed" if self.worker_reply else "failed",
                    "response": self.worker_reply
                    or "The work stopped before it finished.",
                }
            ),
            "list_team": lambda: _dump({"experts": self.world.experts}),
            "list_workspace_files": lambda: _dump(
                {"files": [f.get("name") for f in self._items("drive")]}
            ),
            "read_workspace_file": lambda: _dump(
                {"items": self._search("drive", _words(args))[:1]}
            ),
        }.get(name)
        if handler is not None:
            return handler()
        if name in ("delegate_to_expert", "consult_teammate", "run_sub_session"):
            return _dump({"status": "running", "note": "The work was handed over."})
        return _dump({"status": "ok"})

    def _start_task(self, args: dict[str, Any]) -> str:
        existing = next((t for t in self.tasks if t.id == args.get("task_id")), None)
        if existing is not None:
            # Running an existing task again opens no new one.
            existing.status = "IN_PROGRESS"
            return _dump(
                {
                    "type": "task_started",
                    "task": existing.model_dump(),
                    "in_background": True,
                    "message": f"Started '{existing.title}' again in the background.",
                }
            )
        title = " ".join(str(args.get("title", "")).split())
        duplicate = next(
            (
                t
                for t in self.tasks
                if t.title.casefold() == title.casefold()
                and t.status in ("TODO", "IN_PROGRESS", "NEEDS_INPUT")
            ),
            None,
        )
        if duplicate is not None:
            return _dump(
                {
                    "message": "That work is already running as a task, so no "
                    "second one was started. Tell the user it is under way.",
                    "type": "task_started",
                    "task": duplicate.model_dump(),
                    "in_background": True,
                }
            )
        task = self._add_task(title, "IN_PROGRESS", args.get("expert_id"))
        task.goal = str(args.get("goal") or "") or None
        task.done_when = str(args.get("done_when") or "") or None
        return _dump(
            {
                "type": "task_started",
                "task": task.model_dump(),
                "in_background": True,
                "message": f"Started '{task.title}' in the background; the result "
                "comes back to this chat. If you have not told the user yet, tell "
                "them in one short line for everything you started (what, and "
                "when it comes back; no findings and nothing from the brief); if "
                "you have, write nothing more. Do none of the task here.",
            }
        )

    def _create_task(self, args: dict[str, Any]) -> str:
        task = self._add_task(str(args.get("title", "")), "TODO", args.get("expert_id"))
        return _dump({"type": "task_created", "task": task.model_dump()})

    def _add_task(self, title: str, status: str, owner: Any = None) -> WorldTask:
        task = WorldTask(
            id=f"task_new_{len(self.tasks) + 1}",
            title=title,
            status=status,
            owner=str(owner) if owner else None,
        )
        self.tasks.append(task)
        return task

    def _get_task(self, task_id: str) -> str:
        task = next((t for t in self.tasks if t.id == task_id), None)
        if task is None:
            return _dump({"error": f"Task {task_id} not found"})
        return _dump({"task": task.model_dump()})

    def _update_task(self, args: dict[str, Any]) -> str:
        task = next((t for t in self.tasks if t.id == args.get("task_id")), None)
        if task is None:
            return _dump({"error": f"Task {args.get('task_id')} not found"})
        task.status = str(args.get("status") or task.status)
        task.title = str(args.get("title") or task.title)
        task.goal = str(args.get("goal") or "") or task.goal
        task.done_when = str(args.get("done_when") or "") or task.done_when
        if args.get("assign_to_otto"):
            task.owner = None
        elif args.get("expert_id"):
            task.owner = str(args["expert_id"])
        return _dump({"task": task.model_dump()})

    def _list_tasks(self, args: dict[str, Any]) -> str:
        """The tasks, filtered by status and owner as the real tool filters."""
        status, owner = args.get("status"), args.get("expert_id")
        tasks = [
            t
            for t in self.tasks
            if (not status or t.status == status) and (not owner or t.owner == owner)
        ]
        return _dump({"tasks": [t.model_dump() for t in tasks]})

    # -- data -----------------------------------------------------------------

    def _web_search(self, query: str) -> str:
        words = _words(query)
        entries = self.world.web.get("search", [])
        ranked = sorted(entries, key=lambda e: -_hits(e, words))
        results = [
            r for e in ranked[:2] if _hits(e, words) for r in e.get("results", [])
        ]
        return _dump({"query": query, "results": results})

    def _web_fetch(self, url: str) -> str:
        page = self.world.web.get("pages", {}).get(url)
        return _dump({"url": url, "text": page} if page else {"error": "404 Not Found"})

    def _search(
        self, provider: str, words: list[str], args: Any = None
    ) -> list[dict[str, Any]]:
        items = self._items(provider)
        if days := _days(args):
            # A date or range in the call ("tomorrow's events") picks by date;
            # keyword ranking would return whatever mentions the year.
            items = [item for item in items if _on_days(item, days)]
            return [_clip(item) for item in items[:_MAX_ITEMS]]
        if words:
            # Best match first and, among equal matches, newest first, as a
            # mail search lists them.
            scored = [(_hits(item, words), _when(item), item) for item in items]
            scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
            items = [item for hits, _, item in scored if hits]
        return [_clip(item) for item in items[:_MAX_ITEMS]]

    def _items(self, provider: str) -> list[dict[str, Any]]:
        """Every record of a provider, flattened (Slack messages carry their
        channel, GitHub issues their repo)."""
        data = self.world.data.get(provider) or {}
        items: list[dict[str, Any]] = []
        for key, value in data.items() if isinstance(data, dict) else []:
            for record in value if isinstance(value, list) else []:
                if not isinstance(record, dict):
                    continue
                nested = [v for v in record.values() if _is_records(v)]
                if nested and key in ("channels", "repos"):
                    items += [
                        {**child, key[:-1]: record.get("name")} for child in nested[0]
                    ]
                else:
                    items.append({"kind": key, **record})
        return items


# Production's gate holds a send, post or change for the user's approval in
# these modes (``copilot/gate``); the simulator holds every service write.
_GATED_MODES: frozenset[str] = frozenset({"auto", "ask_first"})


def _failed(result: str) -> bool:
    """The platform answered with an error response."""
    try:
        parsed = json.loads(result)
    except ValueError:
        return False
    return isinstance(parsed, dict) and (
        parsed.get("type") == "error" or "error" in parsed
    )


def _inner(call: ToolCall) -> dict[str, Any]:
    """The arguments of the tool that ran, behind ``run_capability`` or not."""
    inner = call.arguments.get("input") if call.service else call.arguments
    return inner if isinstance(inner, dict) else {}


def _approval_required(cap_id: str) -> str:
    return _dump(
        {
            "type": "approval_required",
            "review_id": f"rev_{abs(hash(cap_id)) % 10**6}",
            "message": "Held for the user's approval: they see the concrete "
            "action on a card. Nothing has run. Don't retry it or get the same "
            "effect another way; carry on with what doesn't depend on it.",
        }
    )


_DATE = re.compile(r"\b(20\d\d-\d\d-\d\d)")


def _days(args: Any) -> list[str]:
    """Every day a call asks about: its dates, and the days between two."""
    found = sorted(set(_DATE.findall(json.dumps(args, default=str)))) if args else []
    if len(found) < 2:
        return found
    first, last = date.fromisoformat(found[0]), date.fromisoformat(found[-1])
    span = min((last - first).days, 31)
    return [(first + timedelta(days=i)).isoformat() for i in range(span + 1)]


def _on_days(item: dict[str, Any], days: list[str]) -> bool:
    stamps = [v for k, v in item.items() if isinstance(v, str) and _DATE.match(v)]
    return any(stamp[:10] in days for stamp in stamps)


def provider_of(text: str) -> str | None:
    lowered = text.lower()
    for provider, aliases in PROVIDER_ALIASES.items():
        if provider in lowered:
            return provider
    for provider, aliases in PROVIDER_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return provider
    return None


def _is_write(cap_id: str, args: dict[str, Any]) -> bool:
    action = cap_id.split("/", 1)[-1].lower() + " " + str(args.get("action", ""))
    return any(word in action for word in _WRITE_WORDS)


def _is_records(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and isinstance(value[0], dict)


# Search operators ("from:gareth") name a field, not a word to find; the
# value after them still is one.
_OPERATOR = re.compile(
    r"\b(?:from|to|cc|bcc|subject|label|in|has|is|after|before|newer_than|"
    r"older_than|filename):",
    re.IGNORECASE,
)
_NOT_WORDS = frozenset({"and", "not"})


def _words(value: Any) -> list[str]:
    """The search words in a query or a call's arguments (their values only:
    the argument names and search operators are not search terms)."""
    text = _OPERATOR.sub(" ", " ".join(_strings(value)))
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in _NOT_WORDS][:20]


def _when(item: Any) -> str:
    """The item's timestamp as sortable text, or "" when it has none."""
    if not isinstance(item, dict):
        return ""
    return str(item.get("date") or item.get("ts") or item.get("start") or "")


def _strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return [str(value)] if value is not None else []


def _hits(item: Any, words: list[str]) -> int:
    """How many of the words the item holds as whole words; one holding all
    of them ranks above every partial match, as a mail search's AND does."""
    found = set(re.findall(r"[a-z0-9]+", " ".join(_strings(item)).lower()))
    hits = sum(1 for w in words if w in found)
    return hits + len(words) if words and hits == len(words) else hits


def _clip(item: dict[str, Any]) -> dict[str, Any]:
    return {k: v[:_ITEM_CHARS] if isinstance(v, str) else v for k, v in item.items()}


def _sign_in(provider: str) -> str:
    return _dump(
        {
            "type": "setup_requirements",
            "provider": provider,
            "message": f"{provider.title()} is not connected. The user sees a "
            "sign-in card in the chat; show it and stop.",
        }
    )


def _dump(value: Any) -> str:
    return json.dumps(value, default=str)[:40_000]
