"""The task record as the simulated world keeps it: what a worker writes on
its task. It applies production's own rules (the caps and the loose matching
in ``data.task_text``), so a line production would refuse is refused here and
proof production could not verify stays unverified here."""

import json
from typing import Any

from backend.copilot.tools.task_progress import FINISHED, NO_TASK
from backend.data.task_text import (
    CLAIM_MAX_CHARS,
    PROGRESS_MAX_CHARS,
    SUMMARY_MAX_CHARS,
    file_key,
    recorded_tool_names,
    too_long,
    tool_names,
    url_key,
)

_PAGE_READS = ("web_fetch", "browser_navigate")


class RecordSim:
    """One chat's task record: the files it saved, the tools it ran, the
    pages it read and the result it left."""

    def __init__(self) -> None:
        self.working = False
        self.files: dict[str, tuple[str, str]] = {}
        self.tools: set[str] = set()
        self.pages: set[str] = set()
        self.finished: dict[str, Any] | None = None

    def note_call(self, name: str, args: dict[str, Any]) -> None:
        """Remember a call that ran, as production's recorder does."""
        self.tools |= recorded_tool_names(name)
        if name in _PAGE_READS and (page := url_key(str(args.get("url") or ""))):
            self.pages.add(page)

    def write_file(self, args: dict[str, Any]) -> str:
        name = _said(args, "filename", "path", "name")
        if not name:
            return _error("A file needs a filename.")
        file_id = f"file_{len(self.files) + 1}"
        self.files[file_key(name)] = (name.rsplit("/", 1)[-1], file_id)
        return _dump(
            {
                "type": "workspace_file_written",
                "file_id": file_id,
                "name": name.rsplit("/", 1)[-1],
                "download_url": f"workspace://{file_id}",
            }
        )

    def report_progress(self, args: dict[str, Any]) -> str:
        line = _said(args, "status_text", "message", "text", "status")
        if not line:
            return _error("Say what you are doing now in status_text.")
        if refusal := too_long(line, PROGRESS_MAX_CHARS, "progress line"):
            return _error(refusal)
        if not self.working:
            return _error(NO_TASK)
        return _dump({"type": "task_progress", "message": "Noted on the task."})

    def add_evidence(self, args: dict[str, Any]) -> str:
        claim = _said(args, "claim", "description", "summary", "text")
        if not claim:
            return _error("Say what happened in claim.")
        if refusal := too_long(claim, CLAIM_MAX_CHARS, "claim"):
            return _error(refusal)
        if not self.working:
            return _error(NO_TASK)
        verified = (
            file_key(_said(args, "file")) in self.files
            or bool(tool_names(_said(args, "tool")) & self.tools)
            or url_key(_said(args, "url")) in self.pages
        )
        return _dump(
            {
                "type": "task_evidence_added",
                "verified": verified,
                "message": (
                    "Proof attached and verified."
                    if verified
                    else "Kept as unverified: nothing on this task's record "
                    "matches. Name a file this task saved, a tool it ran or a "
                    "page it read, or leave it unverified."
                ),
            }
        )

    def finish_task(self, args: dict[str, Any]) -> str:
        summary = _said(args, "summary", "result", "message", "text")
        file = _said(args, "file", "filename", "document", "path")
        if not summary:
            return _error("Say what was done in summary.")
        if refusal := too_long(summary, SUMMARY_MAX_CHARS, "summary"):
            return _error(f"{refusal} The detail belongs in the file.")
        if not self.working:
            return _error(NO_TASK)
        saved = self.files.get(file_key(file)) if file else None
        if file and saved is None:
            return _error(
                f"This task saved no file named '{file}', so nothing was "
                "finished. Save the full result with write_workspace_file, then "
                "call finish_task with that file's name; leave file out when "
                "there is no document."
            )
        self.finished = {"summary": summary, "file": saved[0] if saved else None}
        return _dump(
            {
                "type": "task_finished",
                "document": saved[0] if saved else None,
                "message": FINISHED,
            }
        )


def _said(args: dict[str, Any], *names: str) -> str:
    return next(
        (
            value.strip()
            for name in names
            if isinstance(value := args.get(name), str) and value.strip()
        ),
        "",
    )


def _error(message: str) -> str:
    return _dump({"type": "error", "message": message})


def _dump(value: Any) -> str:
    return json.dumps(value, default=str)
