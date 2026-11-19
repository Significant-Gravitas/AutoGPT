"""Read the event a task should start on ("github", "pull_request.opened",
{"repo": "acme/web"}) as one of the trigger blocks the platform already has.

Trigger blocks say which events a service offers (``webhook_config``: the
provider, an event filter whose fields are the actions, and how an event is
named, e.g. ``pull_request.{event}``) and what they need (a repo). Any
provider with a trigger block and a webhooks manager works; one without gets
a ValueError written for the model, which offers a repeating check instead.
"""

from typing import Any

from pydantic import BaseModel

from backend.blocks import get_block, get_webhook_block_ids
from backend.blocks._base import AnyBlockSchema, BlockType, BlockWebhookConfig
from backend.data.task_text import clip
from backend.integrations.webhooks import supports_webhooks

LINE_MAX_CHARS = 80
_EVENT_WORDS = {"pull_request": "PR", "issues": "Issue"}
# Trigger inputs a task never takes from the model: filled per event, or a
# secret the watch must not keep.
_NOT_CONFIG = {"payload", "secret_token", "constants"}
_SHOWN_ACTIONS = 6


class EventTrigger(BaseModel):
    """A trigger block picked for an event, and the inputs to set it up."""

    block_id: str
    trigger_config: dict[str, Any]
    line: str
    manual: bool


def resolve_event_trigger(
    provider: str, event: str, config: dict[str, Any]
) -> EventTrigger:
    """The trigger block for ``event`` on ``provider`` with its inputs; the
    credentials are added by the caller. Raises ValueError for the model."""
    provider = provider.strip().lower()
    blocks = _provider_blocks(provider)
    if not blocks:
        raise ValueError(
            f"There is no trigger for {provider or 'that service'}, so a task "
            "cannot start on its events. Offer a repeating check instead: "
            "start_task with repeat_crons, e.g. every 15 minutes."
        )
    block, actions = _pick(blocks, event.strip())
    cleaned = _config(block, config)
    filter_input = (
        block.webhook_config.event_filter_input if block.webhook_config else ""
    )
    if filter_input:
        cleaned[filter_input] = {action: True for action in actions}
    manual = block.block_type == BlockType.WEBHOOK_MANUAL
    return EventTrigger(
        block_id=block.id,
        trigger_config=cleaned,
        line=_line(block, actions, cleaned, manual),
        manual=manual,
    )


def _provider_blocks(provider: str) -> list[AnyBlockSchema]:
    blocks = [get_block(block_id) for block_id in get_webhook_block_ids()]
    return [
        block
        for block in blocks
        if block is not None
        and not block.disabled
        and block.webhook_config is not None
        and block.webhook_config.provider.value == provider
        and supports_webhooks(block.webhook_config.provider)
    ]


def _pick(blocks: list[AnyBlockSchema], event: str) -> tuple[AnyBlockSchema, list[str]]:
    """The block whose event names cover ``event``; its actions asked for."""
    for block in blocks:
        names = _actions(block)
        if names is None:
            return block, []
        prefix = _prefix(block)
        if not event.startswith(prefix):
            continue
        wanted = [a.strip() for a in event[len(prefix) :].split(",") if a.strip()]
        if wanted and all(action in names for action in wanted):
            return block, wanted
    offered = "; ".join(
        f"{_prefix(block)}<{'|'.join((_actions(block) or [])[:_SHOWN_ACTIONS])}…>"
        for block in blocks
    )
    raise ValueError(f"Unknown event {event!r}. Events offered: {offered}")


def _actions(block: AnyBlockSchema) -> list[str] | None:
    """The actions the block's event filter offers; None when it has none."""
    config = block.webhook_config
    if config is None or not config.event_filter_input:
        return None
    model = block.input_schema.model_fields[config.event_filter_input].annotation
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        return []
    return list(model.model_fields)


def _prefix(block: AnyBlockSchema) -> str:
    config = block.webhook_config
    return config.event_format.split("{event}")[0] if config else ""


def _config(block: AnyBlockSchema, config: dict[str, Any]) -> dict[str, Any]:
    """The inputs the block needs, from ``config``; ValueError naming any
    required one that is missing, so the model asks the user."""
    schema = block.input_schema
    skipped = _NOT_CONFIG | set(schema.get_credentials_fields())
    if block.webhook_config and block.webhook_config.event_filter_input:
        skipped.add(block.webhook_config.event_filter_input)
    fields = {
        name: info for name, info in schema.model_fields.items() if name not in skipped
    }
    missing = [
        f"{name} ({(info.description or name).splitlines()[0].rstrip('.')})"
        for name, info in fields.items()
        if info.is_required() and config.get(name) in (None, "")
    ]
    if missing:
        raise ValueError(
            "Ask the user for what this trigger needs, then call again with "
            f"it in on_event.config: {'; '.join(missing)}. Do not guess."
        )
    return {name: config[name] for name in fields if name in config}


def _line(
    block: AnyBlockSchema, actions: list[str], config: dict[str, Any], manual: bool
) -> str:
    """What the task watches, in words: "PR opened on acme/web"."""
    if manual:
        return "Calls to its webhook URL"
    kind = _prefix(block).rstrip(".")
    noun = _EVENT_WORDS.get(kind, kind.replace("_", " ").capitalize())
    verbs = " or ".join(action.replace("_", " ") for action in actions)
    resource = _resource(block, config)
    words = " ".join(part for part in (noun, verbs) if part)
    return clip(f"{words} on {resource}" if resource else words, LINE_MAX_CHARS) or kind


def _resource(block: AnyBlockSchema, config: dict[str, Any]) -> str:
    webhook_config = block.webhook_config
    if not isinstance(webhook_config, BlockWebhookConfig):
        return ""
    try:
        return webhook_config.resource_format.format(**config)
    except (KeyError, IndexError):
        return ""
