"""Teams bot commands — /setup, /help, /unlink.

Policy comes from the shared ``command_core``; this module owns only Teams'
transport. Teams has no first-class slash-command surface for bots: commands
arrive as ordinary message text (optionally after an @mention of the bot,
which the adapter has already stripped), and replies render as a normal
message with an Adaptive Card button.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel

from backend.copilot.bot.bot_backend import BotBackend
from backend.copilot.bot.command_core import (
    AdminCheck,
    CommandReply,
    admin_only_reply,
    setup_reply,
    unlink_reply,
)

if TYPE_CHECKING:
    from .adapter import TeamsAdapter

logger = logging.getLogger(__name__)

_HELP_TEXT = (
    "**AutoGPT for Teams**\n"
    "- Add me to a team, then @mention me to run a command: a bare /setup\n"
    "  in a channel never reaches me, but `@AutoGPT /setup` links the team\n"
    "  (team owners only).\n"
    "- @mention me in a channel to chat; I'll reply in a thread.\n"
    "- Message me directly to chat with your personal AutoGPT account.\n"
    "- Run /unlink to manage your linked account and teams."
)

_KNOWN_COMMANDS = {"setup", "help", "unlink"}

# Who may link a team, as the refusal names them.
_ADMIN_ROLE = "team owner"

# The owner check fails mostly when the app lacks its Graph permission (see
# graph.py), which trying again won't fix.
_OWNER_CHECK_HINT = (
    "If this keeps happening, ask your Teams admin to update the AutoGPT app "
    "so it can read this team's members."
)

_PRIVATE_CHAT_NEEDED = (
    "I couldn't send you the setup link in a private chat. Start a chat with "
    "me first, then run `@AutoGPT /setup` here again."
)

_LINK_SENT_PRIVATELY = (
    "I've sent you a private message with the link to finish setting up "
    "AutoGPT for this team."
)


class _TeamDetails(BaseModel):
    name: str = ""
    group_id: str = ""


def parse_command(text: str) -> Optional[str]:
    """Return the command name when ``text`` is one of ours, else None.

    The adapter strips the bot's @mention before this runs, so a channel
    message reads as a bare ``/setup`` exactly like a DM does.
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    command = stripped.split()[0][1:].casefold()
    return command if command in _KNOWN_COMMANDS else None


async def handle(
    api: BotBackend,
    adapter: "TeamsAdapter",
    activity: dict[str, Any],
    command: str,
) -> None:
    """Run ``command`` and render its reply into the conversation."""
    conversation = activity.get("conversation") or {}
    conversation_id = conversation.get("id") or ""
    if not conversation_id:
        return

    if command == "help":
        reply = CommandReply(text=_HELP_TEXT)
    elif command == "unlink":
        reply = unlink_reply()
    else:
        reply = await _setup(api, adapter, activity, conversation_id)

    if reply.button_label and reply.button_url:
        await adapter.send_link(
            conversation_id, reply.text, reply.button_label, reply.button_url
        )
    else:
        await adapter.send_message(conversation_id, reply.text)


async def _setup(
    api: BotBackend,
    adapter: "TeamsAdapter",
    activity: dict[str, Any],
    conversation_id: str,
) -> CommandReply:
    """Link the team this command came from, for one of its owners.

    Teams' unit of installation is the team, so /setup in a channel links the
    team. In a personal chat there is no team to link — the DM link is made
    from the Bots settings page instead. The whole team reads the channel, so
    the link itself goes to the owner in a private chat.
    """
    channel_data = activity.get("channelData") or {}
    team = channel_data.get("team") or {}
    team_id = team.get("id")
    if not team_id:
        return CommandReply(
            text=(
                "Run /setup from a channel in the team you want to link. "
                "To chat with me privately, link your account from the Bots "
                "page in AutoGPT settings."
            )
        )

    sender = activity.get("from") or {}
    details = await _team_details(adapter, activity, team)
    admin_check = await _team_owner_check(
        adapter, details.group_id, sender.get("aadObjectId")
    )
    refusal = admin_only_reply(
        admin_check,
        server_noun="team",
        admin_role=_ADMIN_ROLE,
        retry_hint=_OWNER_CHECK_HINT,
    )
    if refusal is not None:
        return refusal
    private_chat = await _open_private_chat(adapter, sender.get("id") or "")
    if not private_chat:
        return CommandReply(text=_PRIVATE_CHAT_NEEDED)
    reply = await setup_reply(
        api,
        platform="teams",
        server_noun="team",
        admin_check=admin_check,
        admin_role=_ADMIN_ROLE,
        platform_server_id=team_id,
        platform_user_id=sender.get("id") or "",
        platform_username=sender.get("name") or "",
        server_name=details.name,
        channel_id=conversation_id,
    )
    return await _send_privately(adapter, private_chat, reply)


async def _team_details(
    adapter: "TeamsAdapter", activity: dict[str, Any], team: dict[str, Any]
) -> _TeamDetails:
    """The team's display name and Microsoft 365 group id.

    Teams stamps these onto install and conversation-update activities but
    not reliably onto the message that carries /setup, so whatever is missing
    is asked of the Connector. Without the name the settings page can only
    show the raw thread id; without the group id the owner check can't run.
    """
    known = _TeamDetails(
        name=team.get("name") or "", group_id=team.get("aadGroupId") or ""
    )
    service_url = activity.get("serviceUrl") or ""
    team_id = team.get("id") or ""
    if (known.name and known.group_id) or not service_url or not team_id:
        return known
    try:
        fetched = await adapter.client.get_team_details(service_url, team_id) or {}
    except Exception:
        logger.warning("Could not resolve the Teams team details", exc_info=True)
        return known
    return _TeamDetails(
        name=known.name or fetched.get("name") or "",
        group_id=known.group_id or fetched.get("aadGroupId") or "",
    )


async def _team_owner_check(
    adapter: "TeamsAdapter", group_id: str, user_object_id: str | None
) -> AdminCheck:
    """Ask Graph whether the sender owns the team (see ``graph.py``)."""
    if not group_id or not user_object_id:
        return AdminCheck.UNVERIFIED
    try:
        is_owner = await adapter.graph.is_team_owner(group_id, user_object_id)
    except Exception:
        logger.warning("Couldn't check Teams team ownership for /setup", exc_info=True)
        return AdminCheck.UNVERIFIED
    return AdminCheck.ADMIN if is_owner else AdminCheck.NOT_ADMIN


async def _open_private_chat(adapter: "TeamsAdapter", user_id: str) -> str | None:
    try:
        return await adapter.open_dm_channel(user_id)
    except Exception:
        logger.warning("Couldn't open a Teams private chat for /setup", exc_info=True)
        return None


async def _send_privately(
    adapter: "TeamsAdapter", private_chat: str, reply: CommandReply
) -> CommandReply:
    """Send the setup link to the owner's private chat; return what the
    channel sees instead."""
    if not (reply.button_label and reply.button_url):
        return reply  # Already linked, or no link could be made.
    try:
        await adapter.send_link(
            private_chat, reply.text, reply.button_label, reply.button_url
        )
    except Exception:
        logger.warning("Couldn't send the Teams setup link privately", exc_info=True)
        return CommandReply(text=_PRIVATE_CHAT_NEEDED)
    return CommandReply(text=_LINK_SENT_PRIVATELY)
