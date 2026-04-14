"""
Coding Agent Notifications Block — sends task completion notifications to
Discord and Slack, and logs entries to the personal task journal.

Features:
- Discord webhook notifications with rich embeds
- Slack webhook notifications with Block Kit formatting
- Personal task journal: searchable log of every completed task
- Webhook triggers from GitHub or Zapier
"""

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

logger = logging.getLogger(__name__)

DEFAULT_JOURNAL_FILE = "./data/task_journal.json"


class NotificationOperation(str, Enum):
    NOTIFY_DISCORD = "notify_discord"
    NOTIFY_SLACK = "notify_slack"
    LOG_JOURNAL = "log_journal"
    SEARCH_JOURNAL = "search_journal"
    LIST_JOURNAL = "list_journal"
    NOTIFY_ALL = "notify_all"


class TaskOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class NotificationsInput(BlockSchemaInput):
    operation: NotificationOperation = SchemaField(
        default=NotificationOperation.NOTIFY_ALL,
        description="Operation: notify Discord/Slack, log to journal, or search journal.",
    )
    task_title: str = SchemaField(
        default="",
        description="Task title for notifications and journal.",
    )
    task_summary: str = SchemaField(
        default="",
        description="Brief summary of what the agent did.",
    )
    outcome: TaskOutcome = SchemaField(
        default=TaskOutcome.SUCCESS,
        description="Task outcome: success, failure, partial, or cancelled.",
    )
    model_mode: str = SchemaField(
        default="",
        description="Model mode used (standard/max) — included in notification.",
    )
    persona: str = SchemaField(
        default="",
        description="Agent persona used — included in notification.",
    )
    tokens_used: int = SchemaField(
        default=0,
        description="Total tokens consumed by the task.",
    )
    execution_time_secs: int = SchemaField(
        default=0,
        description="Task execution time in seconds.",
    )
    files_changed: list = SchemaField(
        default_factory=list,
        description="List of files changed by the agent.",
    )
    commit_hash: str = SchemaField(
        default="",
        description="Git commit hash if auto-committed.",
    )
    discord_webhook_url: str = SchemaField(
        default="",
        description="Discord webhook URL (overrides settings if provided).",
    )
    slack_webhook_url: str = SchemaField(
        default="",
        description="Slack webhook URL (overrides settings if provided).",
    )
    journal_file: str = SchemaField(
        default=DEFAULT_JOURNAL_FILE,
        description="Path to the task journal JSON file.",
    )
    search_query: str = SchemaField(
        default="",
        description="Search query for SEARCH_JOURNAL operation.",
    )
    journal_limit: int = SchemaField(
        default=20,
        description="Maximum number of journal entries to return.",
    )
    task_id: str = SchemaField(
        default="",
        description="Unique task ID for journal entries.",
    )
    tags: list = SchemaField(
        default_factory=list,
        description="Tags for the journal entry.",
    )


class NotificationsOutput(BlockSchemaOutput):
    discord_sent: bool = SchemaField(description="True if Discord notification was sent.")
    slack_sent: bool = SchemaField(description="True if Slack notification was sent.")
    journal_logged: bool = SchemaField(description="True if journal entry was created.")
    journal_entries: list = SchemaField(description="Journal entries (for search/list operations).")
    status: str = SchemaField(description="Operation result status.")


def _load_journal(journal_file: str) -> list:
    path = Path(journal_file)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _save_journal(journal_file: str, entries: list) -> None:
    path = Path(journal_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, default=str))


def _send_discord(webhook_url: str, title: str, summary: str, outcome: str,
                  model_mode: str, persona: str, tokens: int, commit: str,
                  files: list, exec_time: int) -> tuple[bool, str]:
    """Send a Discord webhook notification with an embed."""
    try:
        import requests
    except ImportError:
        return False, "requests not installed."

    color = {"success": 0x00FF88, "failure": 0xFF4444, "partial": 0xFFAA00, "cancelled": 0x888888}.get(outcome, 0x888888)
    fields = []
    if model_mode:
        fields.append({"name": "Model Mode", "value": model_mode.title(), "inline": True})
    if persona:
        fields.append({"name": "Persona", "value": persona.replace("_", " ").title(), "inline": True})
    if tokens:
        fields.append({"name": "Tokens Used", "value": f"{tokens:,}", "inline": True})
    if exec_time:
        fields.append({"name": "Duration", "value": f"{exec_time}s", "inline": True})
    if commit:
        fields.append({"name": "Commit", "value": f"`{commit}`", "inline": True})
    if files:
        files_str = "\n".join(f"• `{f}`" for f in files[:10])
        if len(files) > 10:
            files_str += f"\n• ...and {len(files) - 10} more"
        fields.append({"name": f"Files Changed ({len(files)})", "value": files_str, "inline": False})

    payload = {
        "embeds": [{
            "title": f"{'✅' if outcome == 'success' else '❌' if outcome == 'failure' else '⚠️'} {title}",
            "description": summary[:2000] if summary else "Task completed.",
            "color": color,
            "fields": fields,
            "footer": {"text": "AutoGPT-V2 Coding Agent"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]
    }

    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        return resp.status_code in (200, 204), f"HTTP {resp.status_code}"
    except Exception as e:
        return False, str(e)


def _send_slack(webhook_url: str, title: str, summary: str, outcome: str,
                model_mode: str, persona: str, tokens: int, commit: str) -> tuple[bool, str]:
    """Send a Slack webhook notification."""
    try:
        import requests
    except ImportError:
        return False, "requests not installed."

    emoji = {"success": ":white_check_mark:", "failure": ":x:", "partial": ":warning:", "cancelled": ":no_entry:"}.get(outcome, ":robot_face:")
    text = f"{emoji} *{title}*\n{summary[:500] if summary else 'Task completed.'}"
    if model_mode:
        text += f"\n*Mode:* {model_mode.title()}"
    if persona:
        text += f" | *Persona:* {persona.replace('_', ' ').title()}"
    if tokens:
        text += f" | *Tokens:* {tokens:,}"
    if commit:
        text += f" | *Commit:* `{commit}`"

    payload = {"text": text}
    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        return resp.status_code == 200, f"HTTP {resp.status_code}"
    except Exception as e:
        return False, str(e)


class CodingAgentNotificationsBlock(Block):
    """
    Sends task completion notifications to Discord/Slack and logs to the personal task journal.

    The task journal is a searchable JSON log of every completed task with summaries,
    model mode, persona, token usage, and git commit information.
    """

    class Input(NotificationsInput):
        pass

    class Output(NotificationsOutput):
        pass

    def __init__(self):
        super().__init__(
            id="e1f2a3b4-c5d6-7890-efab-123456789012",
            description=(
                "Sends Discord/Slack notifications on task completion and logs to "
                "the personal task journal (searchable history of all agent tasks)."
            ),
            categories={BlockCategory.COMMUNICATION},
            input_schema=CodingAgentNotificationsBlock.Input,
            output_schema=CodingAgentNotificationsBlock.Output,
            test_input={
                "operation": NotificationOperation.LOG_JOURNAL.value,
                "task_title": "Test Task",
                "task_summary": "This is a test.",
                "outcome": TaskOutcome.SUCCESS.value,
                "journal_file": "/tmp/test_journal.json",
            },
            test_output=[
                ("journal_logged", True),
                ("discord_sent", False),
                ("slack_sent", False),
            ],
        )

    def run(self, input_data: Input, *, execution_stats=None, **kwargs) -> BlockOutput:
        discord_sent = False
        slack_sent = False
        journal_logged = False
        status_parts = []

        # Determine webhook URLs (from input or settings)
        from backend.util.settings import Settings
        settings = Settings()
        discord_url = input_data.discord_webhook_url or settings.config.discord_webhook_url
        slack_url = input_data.slack_webhook_url or settings.config.slack_webhook_url

        if input_data.operation in (NotificationOperation.NOTIFY_DISCORD, NotificationOperation.NOTIFY_ALL):
            if discord_url:
                ok, msg = _send_discord(
                    discord_url,
                    input_data.task_title,
                    input_data.task_summary,
                    input_data.outcome.value,
                    input_data.model_mode,
                    input_data.persona,
                    input_data.tokens_used,
                    input_data.commit_hash,
                    input_data.files_changed,
                    input_data.execution_time_secs,
                )
                discord_sent = ok
                status_parts.append(f"Discord: {'sent' if ok else f'failed ({msg})'}")
            else:
                status_parts.append("Discord: no webhook URL configured.")

        if input_data.operation in (NotificationOperation.NOTIFY_SLACK, NotificationOperation.NOTIFY_ALL):
            if slack_url:
                ok, msg = _send_slack(
                    slack_url,
                    input_data.task_title,
                    input_data.task_summary,
                    input_data.outcome.value,
                    input_data.model_mode,
                    input_data.persona,
                    input_data.tokens_used,
                    input_data.commit_hash,
                )
                slack_sent = ok
                status_parts.append(f"Slack: {'sent' if ok else f'failed ({msg})'}")
            else:
                status_parts.append("Slack: no webhook URL configured.")

        if input_data.operation in (NotificationOperation.LOG_JOURNAL, NotificationOperation.NOTIFY_ALL):
            import uuid
            entries = _load_journal(input_data.journal_file)
            entry = {
                "id": input_data.task_id or str(uuid.uuid4())[:12],
                "title": input_data.task_title,
                "summary": input_data.task_summary,
                "outcome": input_data.outcome.value,
                "model_mode": input_data.model_mode,
                "persona": input_data.persona,
                "tokens_used": input_data.tokens_used,
                "execution_time_secs": input_data.execution_time_secs,
                "files_changed": input_data.files_changed,
                "commit_hash": input_data.commit_hash,
                "tags": input_data.tags,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            entries.insert(0, entry)  # Most recent first
            _save_journal(input_data.journal_file, entries)
            journal_logged = True
            status_parts.append("Journal: logged.")

        elif input_data.operation == NotificationOperation.SEARCH_JOURNAL:
            entries = _load_journal(input_data.journal_file)
            query = input_data.search_query.lower()
            if query:
                results = [
                    e for e in entries
                    if query in e.get("title", "").lower()
                    or query in e.get("summary", "").lower()
                    or query in " ".join(e.get("tags", [])).lower()
                    or query in e.get("persona", "").lower()
                    or query in e.get("outcome", "").lower()
                ]
            else:
                results = entries
            results = results[:input_data.journal_limit]
            yield "discord_sent", False
            yield "slack_sent", False
            yield "journal_logged", False
            yield "journal_entries", results
            yield "status", f"Found {len(results)} journal entries matching '{query}'."
            return

        elif input_data.operation == NotificationOperation.LIST_JOURNAL:
            entries = _load_journal(input_data.journal_file)
            results = entries[:input_data.journal_limit]
            yield "discord_sent", False
            yield "slack_sent", False
            yield "journal_logged", False
            yield "journal_entries", results
            yield "status", f"Showing {len(results)} of {len(entries)} journal entries."
            return

        yield "discord_sent", discord_sent
        yield "slack_sent", slack_sent
        yield "journal_logged", journal_logged
        yield "journal_entries", []
        yield "status", " | ".join(status_parts) if status_parts else "No operation performed."
