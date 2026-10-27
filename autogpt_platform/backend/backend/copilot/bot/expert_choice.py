"""Turn a typed ``expert=<name>`` into the expert the turn runs as.

Experts are DM-only: a server turn runs as the server-link owner but anyone
in the server types it, so it stays on Otto. A name that matches nothing, or
more than one expert, gets one short reply and no turn.
"""

import logging

from backend.platform_linking.models import BotExpert

from .adapters.base import MessageContext, PlatformAdapter
from .bot_backend import BotBackend
from .options import TurnOptions

logger = logging.getLogger(__name__)

SERVER_EXPERTS_REPLY = (
    "Experts only work in DMs for now. Message me directly to use one, "
    "or ask again without `expert=` to talk to Otto."
)


async def resolve_turn_options(
    api: BotBackend, ctx: MessageContext, adapter: PlatformAdapter
) -> TurnOptions | None:
    """``ctx.options`` with the expert resolved to an id, or None when the
    user was told why the turn can't run."""
    options = ctx.options
    if options.expert_name is None and options.expert_id is None:
        return options
    if not ctx.is_dm:
        await reply_in_place(ctx, adapter, SERVER_EXPERTS_REPLY)
        return None
    if options.expert_id is not None:
        return options
    try:
        experts = await api.list_user_experts(ctx.platform, ctx.user_id, ctx.server_id)
    except Exception:
        logger.exception("Failed to list experts for user %s", ctx.user_id)
        await reply_in_place(
            ctx, adapter, "Couldn't load your experts right now. Try again soon."
        )
        return None
    name = options.expert_name or ""
    matches = match_experts(experts, name)
    if len(matches) == 1:
        return options.model_copy(
            update={"expert_id": matches[0].id, "expert_name": matches[0].name}
        )
    await reply_in_place(ctx, adapter, no_single_match_reply(name, experts, matches))
    return None


def match_experts(experts: list[BotExpert], name: str) -> list[BotExpert]:
    """Exact (case-insensitive) name matches, else names that start with it."""
    wanted = name.casefold()
    exact = [e for e in experts if e.name.casefold() == wanted]
    if exact:
        return exact
    return [e for e in experts if e.name.casefold().startswith(wanted)]


def no_single_match_reply(
    name: str, experts: list[BotExpert], matches: list[BotExpert]
) -> str:
    if matches:
        return f'More than one expert matches "{name}": {_names(matches)}.'
    if not experts:
        return "You haven't hired any experts yet. Hire one in AutoGPT first."
    return f'No expert called "{name}". Your experts: {_names(experts)}.'


def _names(experts: list[BotExpert]) -> str:
    return ", ".join(f"{e.name} ({e.role})" for e in experts)


async def reply_in_place(
    ctx: MessageContext, adapter: PlatformAdapter, text: str
) -> None:
    """Answer where the message was, without opening a thread for it."""
    if ctx.channel_type == "channel":
        await adapter.send_reply(ctx.channel_id, text, ctx.message_id)
        return
    await adapter.send_message(ctx.channel_id, text)
