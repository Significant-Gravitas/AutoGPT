"""Supabase storage layer for the ABN Co-Navigator."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from autogpt.coaching.auth import hash_password, verify_password
from autogpt.coaching.config import coaching_config
from autogpt.coaching.models import (
    Alert,
    AlertLevel,
    ClientStatus,
    KeyResult,
    MasterKeyResult,
    NavigationStatus,
    Objective,
    OKRStatus,
    Obstacle,
    PastSession,
    SessionSummary,
    UserProfile,
    WeeklyLog,
)


# ── Supabase client ───────────────────────────────────────────────────────────

def _get_client():
    from supabase import create_client  # lazy import
    return create_client(coaching_config.supabase_url, coaching_config.supabase_service_key)


# ── User / Auth ───────────────────────────────────────────────────────────────

def register_user(name: str, email: str, password: str) -> UserProfile:
    """Create a new user with hashed password. Raises ValueError on duplicate email."""
    db = _get_client()
    existing = db.table("user_profiles").select("user_id").eq("email", email).execute()
    if existing.data:
        raise ValueError("Email already registered.")
    uid = str(uuid.uuid4())
    db.table("user_profiles").insert({
        "user_id": uid,
        "name": name,
        "email": email,
        "password_hash": hash_password(password),
    }).execute()
    return UserProfile(user_id=uid, name=name, email=email)


def login_user(email: str, password: str) -> UserProfile:
    """Verify credentials and return UserProfile. Raises ValueError on failure."""
    db = _get_client()
    result = db.table("user_profiles").select("*").eq("email", email).execute()
    if not result.data:
        raise ValueError("Invalid email or password.")
    row = result.data[0]
    stored = row.get("password_hash") or ""
    if not stored or not verify_password(password, stored):
        raise ValueError("Invalid email or password.")
    return UserProfile(user_id=row["user_id"], name=row["name"], email=row["email"])


def google_auth(google_id: str, name: str, email: str) -> UserProfile:
    """Register or log in via Google. Links by google_id first, then email."""
    db = _get_client()
    # Check existing google_id
    by_gid = db.table("user_profiles").select("*").eq("google_id", google_id).execute()
    if by_gid.data:
        row = by_gid.data[0]
        return UserProfile(user_id=row["user_id"], name=row["name"], email=row["email"])
    # Check existing email — link Google to existing account
    by_email = db.table("user_profiles").select("*").eq("email", email).execute()
    if by_email.data:
        row = by_email.data[0]
        db.table("user_profiles").update({"google_id": google_id}).eq("user_id", row["user_id"]).execute()
        return UserProfile(user_id=row["user_id"], name=row["name"], email=row["email"])
    # New user
    uid = str(uuid.uuid4())
    db.table("user_profiles").insert({
        "user_id": uid,
        "name": name,
        "email": email,
        "google_id": google_id,
    }).execute()
    return UserProfile(user_id=uid, name=name, email=email)


def get_user_profile(user_id: str) -> Optional[UserProfile]:
    db = _get_client()
    result = db.table("user_profiles").select("user_id,name,email").eq("user_id", user_id).execute()
    if not result.data:
        return None
    row = result.data[0]
    return UserProfile(**row)


# ── Objectives ────────────────────────────────────────────────────────────────

def get_user_objectives(user_id: str) -> List[Objective]:
    """Return all active objectives with their active key results."""
    db = _get_client()
    obj_rows = (
        db.table("objectives")
        .select("*")
        .eq("user_id", user_id)
        .neq("status", OKRStatus.ARCHIVED.value)
        .order("created_at")
        .execute()
        .data or []
    )
    objectives: List[Objective] = []
    for obj in obj_rows:
        kr_rows = (
            db.table("user_key_results")
            .select("*")
            .eq("objective_id", obj["objective_id"])
            .neq("status", OKRStatus.ARCHIVED.value)
            .order("created_at")
            .execute()
            .data or []
        )
        krs = [
            MasterKeyResult(
                kr_id=kr["kr_id"],
                objective_id=kr["objective_id"],
                description=kr["description"],
                status=OKRStatus(kr["status"]),
                current_pct=kr["current_pct"],
            )
            for kr in kr_rows
        ]
        objectives.append(
            Objective(
                objective_id=obj["objective_id"],
                user_id=obj["user_id"],
                title=obj["title"],
                description=obj.get("description", ""),
                status=OKRStatus(obj["status"]),
                key_results=krs,
            )
        )
    return objectives


def upsert_objective(
    user_id: str,
    title: str,
    description: str = "",
    objective_id: Optional[str] = None,
) -> Objective:
    db = _get_client()
    now = datetime.utcnow().isoformat()
    if objective_id:
        db.table("objectives").update({
            "title": title,
            "description": description,
            "updated_at": now,
        }).eq("objective_id", objective_id).eq("user_id", user_id).execute()
    else:
        objective_id = str(uuid.uuid4())
        db.table("objectives").insert({
            "objective_id": objective_id,
            "user_id": user_id,
            "title": title,
            "description": description,
        }).execute()
    return Objective(objective_id=objective_id, user_id=user_id, title=title, description=description)


def set_objective_status(objective_id: str, status: OKRStatus) -> None:
    db = _get_client()
    db.table("objectives").update({
        "status": status.value,
        "updated_at": datetime.utcnow().isoformat(),
    }).eq("objective_id", objective_id).execute()


# ── Key Results ───────────────────────────────────────────────────────────────

def upsert_master_kr(
    objective_id: str,
    user_id: str,
    description: str,
    current_pct: int = 0,
    kr_id: Optional[str] = None,
) -> MasterKeyResult:
    db = _get_client()
    now = datetime.utcnow().isoformat()
    if kr_id:
        db.table("user_key_results").update({
            "description": description,
            "current_pct": current_pct,
            "updated_at": now,
        }).eq("kr_id", kr_id).execute()
    else:
        kr_id = str(uuid.uuid4())
        db.table("user_key_results").insert({
            "kr_id": kr_id,
            "objective_id": objective_id,
            "user_id": user_id,
            "description": description,
            "current_pct": current_pct,
        }).execute()
    return MasterKeyResult(kr_id=kr_id, objective_id=objective_id, description=description, current_pct=current_pct)


def set_kr_status(kr_id: str, status: OKRStatus) -> None:
    db = _get_client()
    db.table("user_key_results").update({
        "status": status.value,
        "updated_at": datetime.utcnow().isoformat(),
    }).eq("kr_id", kr_id).execute()


# ── OKR change application (from AI session) ──────────────────────────────────

def apply_okr_changes(user_id: str, changes: List[Dict[str, Any]]) -> None:
    """Apply structured OKR mutations extracted from a session conversation."""
    db = _get_client()
    for change in changes:
        action = change.get("action", "")
        try:
            if action == "add_objective":
                upsert_objective(user_id=user_id, title=change["title"], description=change.get("description", ""))
            elif action == "edit_objective":
                upsert_objective(
                    user_id=user_id,
                    title=change["title"],
                    description=change.get("description", ""),
                    objective_id=change.get("objective_id"),
                )
            elif action == "archive_objective":
                set_objective_status(change["objective_id"], OKRStatus.ARCHIVED)
            elif action == "hold_objective":
                set_objective_status(change["objective_id"], OKRStatus.ON_HOLD)
            elif action == "reactivate_objective":
                set_objective_status(change["objective_id"], OKRStatus.ACTIVE)
            elif action == "add_kr":
                upsert_master_kr(
                    objective_id=change["objective_id"],
                    user_id=user_id,
                    description=change["description"],
                    current_pct=change.get("current_pct", 0),
                )
            elif action == "edit_kr":
                upsert_master_kr(
                    objective_id=change.get("objective_id", ""),
                    user_id=user_id,
                    description=change["description"],
                    current_pct=change.get("current_pct", 0),
                    kr_id=change.get("kr_id"),
                )
            elif action == "update_kr_pct":
                db.table("user_key_results").update({
                    "current_pct": int(change["current_pct"]),
                    "updated_at": datetime.utcnow().isoformat(),
                }).eq("kr_id", change["kr_id"]).execute()
            elif action == "archive_kr":
                set_kr_status(change["kr_id"], OKRStatus.ARCHIVED)
            elif action == "hold_kr":
                set_kr_status(change["kr_id"], OKRStatus.ON_HOLD)
            elif action == "reactivate_kr":
                set_kr_status(change["kr_id"], OKRStatus.ACTIVE)
        except Exception:
            # Skip invalid changes rather than failing the whole session
            pass


# ── History ───────────────────────────────────────────────────────────────────

def get_past_sessions(user_id: str, limit: int = 5) -> List[PastSession]:
    """Return the most recent session summaries for a user."""
    db = _get_client()
    rows = (
        db.table("coaching_sessions")
        .select("session_id,timestamp,alert_level,summary_for_coach")
        .eq("user_id", user_id)
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
        .data or []
    )
    return [
        PastSession(
            session_id=r["session_id"],
            timestamp=r["timestamp"],
            alert_level=r["alert_level"],
            summary_for_coach=r["summary_for_coach"],
        )
        for r in rows
    ]


# ── Session save / load ───────────────────────────────────────────────────────

def _ensure_client_exists(db, client_id: str, client_name: str) -> None:
    db.table("clients").upsert(
        {"client_id": client_id, "name": client_name},
        on_conflict="client_id",
    ).execute()


def save_session(summary: SessionSummary) -> None:
    """Persist a SessionSummary to Supabase and apply any OKR changes."""
    db = _get_client()
    _ensure_client_exists(db, summary.client_id, summary.client_name)

    session_row: Dict[str, Any] = {
        "session_id": summary.session_id,
        "client_id": summary.client_id,
        "timestamp": summary.timestamp.isoformat(),
        "focus_goal": summary.weekly_log.focus_goal,
        "environmental_changes": summary.weekly_log.environmental_changes,
        "mood_indicator": summary.weekly_log.mood_indicator,
        "alert_level": summary.alerts.level.value,
        "alert_reason": summary.alerts.reason,
        "summary_for_coach": summary.summary_for_coach,
        "raw_conversation": None,
    }
    if summary.user_id:
        session_row["user_id"] = summary.user_id

    db.table("coaching_sessions").upsert(session_row, on_conflict="session_id").execute()

    db.table("key_results").delete().eq("session_id", summary.session_id).execute()
    if summary.weekly_log.key_results:
        db.table("key_results").insert([
            {
                "session_id": summary.session_id,
                "kr_id": kr.kr_id,
                "description": kr.description,
                "status_pct": kr.status_pct,
                "status_color": kr.status_color,
            }
            for kr in summary.weekly_log.key_results
        ]).execute()

    db.table("obstacles").delete().eq("session_id", summary.session_id).execute()
    if summary.weekly_log.obstacles:
        db.table("obstacles").insert([
            {
                "session_id": summary.session_id,
                "description": obs.description,
                "reported_at": obs.reported_at.isoformat() if obs.reported_at else None,
                "resolved": obs.resolved,
            }
            for obs in summary.weekly_log.obstacles
        ]).execute()

    # Apply OKR mutations requested during the session
    if summary.okr_changes and summary.user_id:
        apply_okr_changes(summary.user_id, summary.okr_changes)


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
        db.table("key_results").select("*").eq("session_id", session_id).execute().data or []
    )
    obs_rows = (
        db.table("obstacles").select("*").eq("session_id", session_id).execute().data or []
    )

    key_results = [
        KeyResult(
            kr_id=r["kr_id"],
            description=r["description"],
            status_pct=r["status_pct"],
            status_color=r.get("status_color", ""),
        )
        for r in kr_rows
    ]

    obstacles = [
        Obstacle(
            description=r["description"],
            reported_at=datetime.fromisoformat(r["reported_at"]) if r.get("reported_at") else None,
            resolved=r["resolved"],
        )
        for r in obs_rows
    ]

    client_row = (
        db.table("clients").select("name").eq("client_id", row["client_id"]).single().execute()
    )
    client_name = client_row.data["name"] if client_row.data else row["client_id"]

    return SessionSummary(
        session_id=row["session_id"],
        client_id=row["client_id"],
        client_name=client_name,
        user_id=row.get("user_id"),
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
    db = _get_client()
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
