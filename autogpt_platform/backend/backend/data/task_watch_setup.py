"""Set up what a task watches: create or reuse the webhook for a trigger
block, the way a workflow trigger gets one (``setup_webhook_for_block``).

A provider webhook (GitHub) is registered under the user's connected account
and reused when one already covers the same resource and events; a manual
webhook (the generic one) is new each time, with a URL the user calls. The
watch keeps the trigger block's inputs without their credentials: the event
filter is all the ingress needs.

When a task stops watching, its webhook is pruned the way a workflow's is when
its trigger goes (``prune_webhook_if_dangling``): only when no workflow
trigger and no other open task uses it. Best-effort, after the task's own
write: a webhook left over is pruned the next time something lets go of it.
"""

import logging
from typing import Any

from pydantic import JsonValue

from backend.blocks import get_block
from backend.blocks._base import AnyBlockSchema, BlockType
from backend.data.integrations import get_webhook
from backend.data.task_schedule_meta import TaskWatch
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.webhooks import get_webhook_manager
from backend.integrations.webhooks.utils import setup_webhook_for_block
from backend.util.exceptions import InvalidInputError, NotFoundError

logger = logging.getLogger(__name__)


async def setup_task_watch(
    user_id: str,
    block_id: str,
    trigger_config: dict[str, JsonValue],
    line: str,
    organization_id: str | None = None,
    team_id: str | None = None,
) -> TaskWatch:
    """The watch for the trigger block ``block_id`` with ``trigger_config``
    (its inputs, credentials included). Raises InvalidInputError, written for
    the model, when the webhook cannot be set up."""
    block = get_block(block_id)
    if block is None or block.webhook_config is None:
        raise NotFoundError(f"Trigger block {block_id} not found")
    webhook, feedback = await setup_webhook_for_block(
        user_id=user_id,
        trigger_block=block,
        trigger_config=trigger_config,
        organization_id=organization_id,
        team_id=team_id,
    )
    if webhook is None:
        raise InvalidInputError(f"Could not set up the webhook: {feedback}")
    manual = block.block_type == BlockType.WEBHOOK_MANUAL
    return TaskWatch(
        webhook_id=webhook.id,
        provider=webhook.provider.value,
        trigger_block_id=block.id,
        trigger_config=without_credentials(block, trigger_config),
        line=line,
        url=webhook.url if manual else None,
    )


def without_credentials(
    block: AnyBlockSchema, trigger_config: dict[str, JsonValue]
) -> dict[str, Any]:
    """The trigger block's inputs with every credentials field taken out."""
    secret = set(block.input_schema.get_credentials_fields())
    return {key: value for key, value in trigger_config.items() if key not in secret}


async def release_watch_webhook(user_id: str, watch: TaskWatch) -> None:
    """Prune the watched webhook if nothing else uses it any more."""
    try:
        webhook = await get_webhook(watch.webhook_id)
        if webhook.user_id != user_id:
            return
        credentials = (
            await IntegrationCredentialsManager().get(
                user_id, webhook.credentials_id, allow_sdk_default=True
            )
            if webhook.credentials_id
            else None
        )
        await get_webhook_manager(webhook.provider).prune_webhook_if_dangling(
            user_id, webhook.id, credentials
        )
    except Exception as e:
        logger.warning(
            f"Best-effort prune of watched webhook #{watch.webhook_id} failed: {e}"
        )
