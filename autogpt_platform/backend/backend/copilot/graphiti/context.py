"""Warm context retrieval — pre-loads relevant facts at session start."""

import asyncio
import logging
from datetime import datetime, timezone

from ._format import (
    extract_episode_body,
    extract_episode_timestamp,
    extract_fact,
    extract_temporal_validity,
)
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
            valid_from, valid_to = extract_temporal_validity(e)
            fact = extract_fact(e)
            fact_lines.append(f"  - {fact} ({valid_from} — {valid_to})")
        sections.append("<FACTS>\n" + "\n".join(fact_lines) + "\n</FACTS>")

    if episodes:
        ep_lines = []
        for ep in episodes:
            ts = extract_episode_timestamp(ep)
            body = extract_episode_body(ep)
            ep_lines.append(f"  - [{ts}] {body}")
        sections.append(
            "<RECENT_EPISODES>\n" + "\n".join(ep_lines) + "\n</RECENT_EPISODES>"
        )

    body = "\n\n".join(sections)
    return f"<temporal_context>\n{body}\n</temporal_context>"
