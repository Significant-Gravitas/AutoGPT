"""Teams bot commands — /setup, /help, /unlink.

Policy comes from the shared ``command_core``; this module owns only Teams'
transport. Teams has no first-class slash-command surface for bots: commands
arrive as ordinary message text (optionally after an @mention of the bot,
which the adapter has already stripped), and replies render as a normal
message with an Adaptive Card button.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

from backend.copilot.bot.bot_backend import BotBackend
from backend.copilot.bot.command_core import CommandReply, setup_reply, unlink_reply

if TYPE_CHECKING:
    from .adapter import TeamsAdapter

logger = logging.getLogger(__name__)

_HELP_TEXT = (
    "**AutoGPT for Teams**\n"
    "- Add me to a team, then @mention me to run a command: a bare /setup\n"
    "  in a channel never reaches me, but `@AutoGPT /setup` links the team.\n"
    "- @mention me in a channel to chat; I'll reply in a thread.\n"
    "- Message me directly to chat with your personal AutoGPT account.\n"
    "- Run /unlink to manage your linked account and teams."
)

_KNOWN_COMMANDS = {"setup", "help", "unlink"}


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
    """Link the team this command came from.

    Teams' unit of installation is the team, so /setup in a channel links the
    team. In a personal chat there is no team to link — the DM link is made
    from the Bots settings page instead.
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
    return await setup_reply(
        api,
        platform="teams",
        server_noun="team",
        platform_server_id=team_id,
        platform_user_id=sender.get("id") or "",
        platform_username=sender.get("name") or "",
        server_name=await _team_name(adapter, activity, team),
        channel_id=conversation_id,
    )


async def _team_name(
    adapter: "TeamsAdapter", activity: dict[str, Any], team: dict[str, Any]
) -> str:
    """The team's display name, asking the Connector when the activity omits it.

    Teams stamps ``channelData.team.name`` onto install and conversation-update
    activities but not onto the message that carries /setup, so relying on the
    activity alone stores an empty name and the settings page can only show the
    raw thread id.
    """
    name = team.get("name") or ""
    if name:
        return name
    service_url = activity.get("serviceUrl") or ""
    team_id = team.get("id") or ""
    if not service_url or not team_id:
        return ""
    try:
        details = await adapter.client.get_team_details(service_url, team_id)
    except Exception:
        logger.warning("Could not resolve the Teams team name", exc_info=True)
        return ""
    return (details or {}).get("name") or ""
