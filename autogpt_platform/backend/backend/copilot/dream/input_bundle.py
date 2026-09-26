"""A dream pass's ``DreamInput`` as plain JSON.

One format, two stores: the batch path persists it to Redis so each phase
callback can rebuild its prompt (``batch_submit.persist_input_bundle``), and
the pass's durable record keeps the same dict (``backend/data/dream_pass.py``).
It round-trips through ``json`` rather than pickling so the wire format stays
portable and debuggable.
"""

from datetime import datetime
from typing import Any

from .fetch import DreamInput, EpisodeRow, FactRow, SessionRow


def input_bundle_to_dict(input_bundle: DreamInput) -> dict[str, Any]:
    return {
        "user_id": input_bundle.user_id,
        "expert_id": input_bundle.expert_id,
        "group_id": input_bundle.group_id,
        "window_start": input_bundle.window_start.isoformat(),
        "window_end": input_bundle.window_end.isoformat(),
        "episodes": [
            {
                "uuid": e.uuid,
                "name": e.name,
                "content": e.content,
                "source_description": e.source_description,
                "valid_at": e.valid_at,
                "created_at": e.created_at,
            }
            for e in input_bundle.episodes
        ],
        "facts": [
            {
                "uuid": f.uuid,
                "source": f.source,
                "target": f.target,
                "name": f.name,
                "fact": f.fact,
                "scope": f.scope,
                "confidence": f.confidence,
                "status": f.status,
                "created_at": f.created_at,
            }
            for f in input_bundle.facts
        ],
        "recent_sessions": [
            {
                "session_id": s.session_id,
                "title": s.title,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "body": s.body,
            }
            for s in input_bundle.recent_sessions
        ],
        "known_fact_uuids": list(input_bundle.known_fact_uuids),
        "known_episode_uuids": list(input_bundle.known_episode_uuids),
    }


def input_bundle_from_dict(data: dict[str, Any]) -> DreamInput:
    return DreamInput(
        user_id=data["user_id"],
        expert_id=data.get("expert_id"),
        group_id=data["group_id"],
        window_start=datetime.fromisoformat(data["window_start"]),
        window_end=datetime.fromisoformat(data["window_end"]),
        episodes=[EpisodeRow(**e) for e in data.get("episodes") or []],
        facts=[FactRow(**f) for f in data.get("facts") or []],
        recent_sessions=[
            SessionRow(
                session_id=s["session_id"],
                title=s.get("title"),
                created_at=(
                    datetime.fromisoformat(s["created_at"])
                    if s.get("created_at")
                    else None
                ),
                body=s.get("body") or "",
            )
            for s in data.get("recent_sessions") or []
        ],
        known_fact_uuids=set(data.get("known_fact_uuids") or []),
        known_episode_uuids=set(data.get("known_episode_uuids") or []),
    )
