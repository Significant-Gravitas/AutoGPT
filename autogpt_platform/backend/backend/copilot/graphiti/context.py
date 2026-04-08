"""Warm context retrieval — pre-loads relevant facts at session start."""

import asyncio
import logging
from datetime import datetime, timezone

from .client import derive_group_id, get_graphiti_client
from .config import graphiti_config

logger = logging.getLogger(__name__)


async def fetch_warm_context(user_id: str, message: str) -> str | None:
    """Fetch relevant temporal context for the current user and message.

    Called at the start of a session (first turn) to pre-load facts from
    prior conversations.  Returns a formatted ``<temporal_context>`` block
    suitable for appending to the system prompt, or ``None`` on failure.

    Graceful degradation: any error (timeout, connection, graphiti-core bug)
    returns ``None`` so the copilot continues without temporal context.
    """
    if not user_id:
        return None

    try:
        return await asyncio.wait_for(
            _fetch(user_id, message),
            timeout=graphiti_config.context_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Graphiti warm context timed out after %.1fs",
            graphiti_config.context_timeout,
        )
        return None
    except Exception:
        logger.warning("Graphiti warm context fetch failed", exc_info=True)
        return None


async def _fetch(user_id: str, message: str) -> str | None:
    group_id = derive_group_id(user_id)
    client = await get_graphiti_client(group_id)

    edges, episodes = await asyncio.gather(
        client.search(
            query=message,
            group_ids=[group_id],
            num_results=graphiti_config.context_max_facts,
        ),
        client.retrieve_episodes(
            reference_time=datetime.now(timezone.utc),
            group_ids=[group_id],
            last_n=5,
        ),
    )

    if not edges and not episodes:
        return None

    return _format_context(edges, episodes)


def _format_context(edges, episodes) -> str:
    sections: list[str] = []

    if edges:
        fact_lines = []
        for e in edges:
            valid_from = getattr(e, "valid_at", "unknown")
            valid_to = getattr(e, "invalid_at", "present")
            fact = getattr(e, "fact", "") or getattr(e, "name", "")
            fact_lines.append(f"  - {fact} ({valid_from} — {valid_to})")
        sections.append("<FACTS>\n" + "\n".join(fact_lines) + "\n</FACTS>")

    if episodes:
        ep_lines = []
        for ep in episodes:
            ts = getattr(ep, "created_at", "")
            body = str(
                getattr(ep, "content", "")
                or getattr(ep, "body", "")
                or getattr(ep, "episode_body", "")
            )[:500]
            ep_lines.append(f"  - [{ts}] {body}")
        sections.append(
            "<RECENT_EPISODES>\n" + "\n".join(ep_lines) + "\n</RECENT_EPISODES>"
        )

    body = "\n\n".join(sections)
    return f"<temporal_context>\n{body}\n</temporal_context>"
