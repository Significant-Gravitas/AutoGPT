"""Supabase storage layer for coaching sessions."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from supabase import Client, create_client

from autogpt.coaching.config import coaching_config
from autogpt.coaching.models import (
    Alert,
    AlertLevel,
    ClientStatus,
    KeyResult,
    NavigationStatus,
    Obstacle,
    SessionSummary,
    WeeklyLog,
)


def _get_client() -> Client:
    return create_client(coaching_config.supabase_url, coaching_config.supabase_service_key)


def _ensure_client_exists(db: Client, client_id: str, client_name: str) -> None:
    """Upsert client record so the foreign key constraint is satisfied."""
    db.table("clients").upsert(
        {"client_id": client_id, "name": client_name},
        on_conflict="client_id",
    ).execute()


def save_session(summary: SessionSummary) -> None:
    """Persist a SessionSummary to Supabase."""
    db = _get_client()
    _ensure_client_exists(db, summary.client_id, summary.client_name)

    # Upsert the session row
    db.table("coaching_sessions").upsert(
        {
            "session_id": summary.session_id,
            "client_id": summary.client_id,
            "timestamp": summary.timestamp.isoformat(),
            "focus_goal": summary.weekly_log.focus_goal,
            "environmental_changes": summary.weekly_log.environmental_changes,
            "mood_indicator": summary.weekly_log.mood_indicator,
            "alert_level": summary.alerts.level.value,
            "alert_reason": summary.alerts.reason,
            "summary_for_coach": summary.summary_for_coach,
            "raw_conversation": None,  # populated separately if needed
        },
        on_conflict="session_id",
    ).execute()

    # Insert key results (delete existing first to avoid duplicates)
    db.table("key_results").delete().eq("session_id", summary.session_id).execute()
    if summary.weekly_log.key_results:
        db.table("key_results").insert(
            [
                {
                    "session_id": summary.session_id,
                    "kr_id": kr.kr_id,
                    "description": kr.description,
                    "status_pct": kr.status_pct,
                    "status_color": kr.status_color,
                }
                for kr in summary.weekly_log.key_results
            ]
        ).execute()

    # Insert obstacles
    db.table("obstacles").delete().eq("session_id", summary.session_id).execute()
    if summary.weekly_log.obstacles:
        db.table("obstacles").insert(
            [
                {
                    "session_id": summary.session_id,
                    "description": obs.description,
                    "reported_at": obs.reported_at.isoformat(),
                    "resolved": obs.resolved,
                }
                for obs in summary.weekly_log.obstacles
            ]
        ).execute()


def load_session(session_id: str) -> Optional[SessionSummary]:
    """Load a session from Supabase by session_id."""
    db = _get_client()

    session_row = (
        db.table("coaching_sessions")
        .select("*")
        .eq("session_id", session_id)
        .single()
        .execute()
    )
    if not session_row.data:
        return None

    row = session_row.data
    kr_rows = (
        db.table("key_results")
        .select("*")
        .eq("session_id", session_id)
        .execute()
        .data
    )
    obs_rows = (
        db.table("obstacles")
        .select("*")
        .eq("session_id", session_id)
        .execute()
        .data
    )

    key_results = [
        KeyResult(
            kr_id=r["kr_id"],
            description=r["description"],
            status_pct=r["status_pct"],
            status_color=r["status_color"],
        )
        for r in (kr_rows or [])
    ]

    obstacles = [
        Obstacle(
            description=r["description"],
            reported_at=datetime.fromisoformat(r["reported_at"]),
            resolved=r["resolved"],
        )
        for r in (obs_rows or [])
    ]

    # Load client name
    client_row = (
        db.table("clients")
        .select("name")
        .eq("client_id", row["client_id"])
        .single()
        .execute()
    )
    client_name = client_row.data["name"] if client_row.data else row["client_id"]

    return SessionSummary(
        session_id=row["session_id"],
        client_id=row["client_id"],
        client_name=client_name,
        timestamp=datetime.fromisoformat(row["timestamp"]),
        weekly_log=WeeklyLog(
            focus_goal=row.get("focus_goal", ""),
            key_results=key_results,
            environmental_changes=row.get("environmental_changes", ""),
            obstacles=obstacles,
            mood_indicator=row.get("mood_indicator", ""),
        ),
        alerts=Alert(
            level=AlertLevel(row["alert_level"]),
            reason=row.get("alert_reason", ""),
        ),
        summary_for_coach=row.get("summary_for_coach", ""),
    )


def get_latest_session_per_client() -> List[SessionSummary]:
    """Return the most recent session for each client."""
    db = _get_client()

    # Get all clients
    clients = db.table("clients").select("client_id, name").execute().data or []
    results = []

    for client in clients:
        latest = (
            db.table("coaching_sessions")
            .select("session_id")
            .eq("client_id", client["client_id"])
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )
        if latest.data:
            summary = load_session(latest.data[0]["session_id"])
            if summary:
                results.append(summary)

    return results


def _navigation_status(avg_pct: float) -> NavigationStatus:
    if avg_pct >= 70:
        return NavigationStatus.CLEAR
    if avg_pct >= 40:
        return NavigationStatus.CHOPPY
    return NavigationStatus.STORMY


def get_client_statuses() -> List[ClientStatus]:
    """Build a ClientStatus list from the latest session per client."""
    sessions = get_latest_session_per_client()
    statuses = []
    for s in sessions:
        avg = s.weekly_log.avg_kr_pct()
        active_alerts = [s.alerts] if s.alerts.level != AlertLevel.GREEN else []
        statuses.append(
            ClientStatus(
                client_id=s.client_id,
                name=s.client_name,
                navigation_status=_navigation_status(avg),
                key_results_avg=round(avg, 1),
                active_alerts=active_alerts,
                last_session=s.timestamp,
            )
        )
    return statuses
