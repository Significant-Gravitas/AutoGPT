"""What a heartbeat turn is told: OpenClaw's heartbeat body, adapted."""

from datetime import datetime

# The whole reply when nothing needs the user. HEARTBEAT_OK is OpenClaw's
# token; a model that learned it is understood too.
NO_REPLY = "NO_REPLY"
SILENT_TOKENS: tuple[str, ...] = (NO_REPLY, "HEARTBEAT_OK")
RESPOND_TOOL = "heartbeat_respond"


def build_prompt(checklist: str, now_local: datetime, tz_name: str) -> str:
    """The turn's only message. The checklist is the user's own text, fenced
    so a line in it cannot pass for this frame."""
    when = now_local.strftime("%A %d %B %Y, %H:%M")
    return (
        "This is a scheduled heartbeat check, not a message from the user. "
        "Nobody is watching this chat, so do not ask questions.\n\n"
        "The user's heartbeat checklist:\n"
        "<heartbeat_checklist>\n"
        f"{checklist.strip()}\n"
        "</heartbeat_checklist>\n\n"
        "1. Work through the checklist strictly. Do only what it asks, and use "
        "only tools that read; change nothing.\n"
        "2. Do not infer or repeat old tasks from prior chats. Only the "
        "checklist and what you find now count.\n"
        "3. Before alerting, call memory_search for what you already told the "
        "user about it recently, and do not alert about the same thing twice.\n"
        "4. If something needs the user's attention now, call "
        f"`{RESPOND_TOOL}` (through run_capability, id `tool:{RESPOND_TOOL}`) "
        "with notify=true and a notification_text of one or two sentences: "
        "what happened and why it matters now.\n"
        f"5. If nothing needs attention, reply exactly {NO_REPLY} and nothing "
        "else.\n\n"
        f"Current time: {when} ({tz_name})."
    )
