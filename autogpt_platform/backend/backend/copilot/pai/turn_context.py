"""Per-turn context for a pai turn, built with the baseline's own builders.

* :func:`first_turn_prefix` - the persisted first-message prefix
  (``inject_user_context`` with budget, session, skills and expert context),
  exactly as the baseline writes it onto the first user row.
* :func:`turn_context_blocks` - the query-only blocks the baseline prepends to
  the live request and never persists; here they become the dynamic
  instructions (see :mod:`.prompt`).
* :func:`build_user_prompt` - the request's user content: the message, the
  attachment and URL hints, queued messages, late held-call results and any
  images, in the baseline's order.
"""

import base64
import logging

from pydantic_ai.messages import BinaryContent, UserContent

from backend.copilot.budget_signal import build_turn_budget_block
from backend.copilot.builder_context import build_builder_context_turn_prefix
from backend.copilot.config import ChatConfig
from backend.copilot.model import ChatSession
from backend.copilot.pending_messages import (
    PendingMessage,
    format_pending_as_user_message,
)
from backend.copilot.rate_limit import build_budget_ctx
from backend.copilot.service import inject_user_context
from backend.copilot.tools.seen_capabilities import build_seen_capabilities_notice
from backend.copilot.tools.session_context import build_session_context
from backend.copilot.tools.skills import (
    build_skills_context,
    build_skills_update_notice,
)
from backend.copilot.tree import TurnEnvelope
from backend.data.understanding import BusinessUnderstanding

logger = logging.getLogger(__name__)


async def first_turn_prefix(
    understanding: BusinessUnderstanding | None,
    message: str,
    session: ChatSession,
    user_id: str | None,
    config: ChatConfig,
) -> str | None:
    """Wrap and persist the first user row, as the baseline does."""
    budget_ctx = await build_budget_ctx(
        user_id=user_id,
        default_daily_cost_limit=config.daily_cost_limit_microdollars,
        default_weekly_cost_limit=config.weekly_cost_limit_microdollars,
    )
    session_ctx = (
        await build_session_context(session_id=session.session_id, user_id=user_id)
        if user_id
        else ""
    )
    try:
        skills_ctx = await build_skills_context(user_id, expert_id=session.expert_id)
    except Exception:
        logger.exception("[PAI] Could not build the skill index; continuing without")
        skills_ctx = ""
    return await inject_user_context(
        understanding,
        message,
        session.session_id,
        session.messages,
        budget_ctx=budget_ctx,
        session_ctx=session_ctx,
        skills_ctx=skills_ctx,
        user_id=user_id,
        expert_id=session.expert_id,
    )


async def turn_context_blocks(
    session: ChatSession,
    *,
    user_id: str | None,
    envelope: TurnEnvelope | None,
    is_user_message: bool,
    warm_ctx: str | None,
) -> list[str]:
    """The baseline's query-only blocks for this turn, in its order."""
    blocks = [await build_turn_budget_block(envelope, user_id), warm_ctx or ""]
    if is_user_message and session.metadata.builder_graph_id:
        blocks.append(await build_builder_context_turn_prefix(session, user_id))
    if is_user_message and user_id:
        blocks.append(await _skills_notice(session, user_id))
        blocks.append(build_seen_capabilities_notice(session))
    return [block for block in blocks if block]


async def _skills_notice(session: ChatSession, user_id: str) -> str:
    try:
        return await build_skills_update_notice(
            user_id,
            expert_id=session.expert_id,
            prior_contents=[
                m.content or "" for m in session.messages if m.role == "user"
            ],
        )
    except Exception:
        logger.exception("[PAI] Could not build the skills update notice")
        return ""


def url_context_hint(context: dict[str, str] | None) -> str:
    """The baseline's hint for a page the user shared."""
    if not context or not context.get("url"):
        return ""
    url = context["url"]
    content = context.get("content", "")
    if content:
        return f"\n[The user shared a URL: {url}\nContent:\n{content[:8000]}]"
    return f"\n[The user shared a URL: {url}]"


def build_user_prompt(
    message: str | None,
    *,
    hint: str,
    queued: list[PendingMessage],
    late_results: list[PendingMessage],
    image_blocks: list[dict],
) -> str | list[UserContent] | None:
    """The request's user content; a list only when there is more than text."""
    text = message or ""
    if hint:
        text = f"{text}\n{hint}" if text else hint
    extra = [
        format_pending_as_user_message(pm)["content"] for pm in [*queued, *late_results]
    ]
    images = [_image(block) for block in image_blocks]
    parts: list[str] = [part for part in [text, *extra] if part]
    if not parts and not images:
        return None
    if len(parts) == 1 and not images:
        return parts[0]
    return [*parts, *images]


def _image(block: dict) -> BinaryContent:
    source = block["source"]
    return BinaryContent(
        data=base64.b64decode(source["data"]), media_type=source["media_type"]
    )
