"""
Database operations for long-running agent sessions.
"""

import logging
from typing import Any, Optional

from prisma import Prisma
from prisma.models import (
    LongRunningFeature,
    LongRunningProgress,
    LongRunningSession,
)

from backend.util.service import get_service_client

from . import model

logger = logging.getLogger(__name__)


async def get_db() -> Prisma:
    """Get database client."""
    return get_service_client().db_client


# === Session Operations ===


async def create_session(
    user_id: str,
    project_name: str,
    project_description: str,
    working_directory: str,
    features: list[dict[str, Any]] = [],
) -> LongRunningSession:
    """Create a new long-running session."""
    db = await get_db()

    session = await db.longrunningsession.create(
        data={
            "userId": user_id,
            "projectName": project_name,
            "projectDescription": project_description,
            "workingDirectory": working_directory,
            "status": "INITIALIZING",
        }
    )

    # Create features if provided
    for i, f in enumerate(features):
        await db.longrunningfeature.create(
            data={
                "sessionId": session.id,
                "featureId": f.get("feature_id", f"feature_{i+1:03d}"),
                "category": f.get("category", "FUNCTIONAL").upper(),
                "description": f.get("description", ""),
                "steps": f.get("steps", []),
                "priority": f.get("priority", 5),
                "dependencies": f.get("dependencies", []),
                "status": "PENDING",
            }
        )

    return session


async def get_session(session_id: str, user_id: str) -> Optional[LongRunningSession]:
    """Get a session by ID."""
    db = await get_db()

    return await db.longrunningsession.find_first(
        where={
            "id": session_id,
            "userId": user_id,
        }
    )


async def get_session_with_details(
    session_id: str, user_id: str
) -> Optional[LongRunningSession]:
    """Get a session with features and recent progress."""
    db = await get_db()

    return await db.longrunningsession.find_first(
        where={
            "id": session_id,
            "userId": user_id,
        },
        include={
            "Features": {
                "order_by": [{"priority": "asc"}, {"createdAt": "asc"}]
            },
            "ProgressLog": {
                "order_by": {"createdAt": "desc"},
                "take": 50,
            },
        },
    )


async def list_sessions(
    user_id: str,
    status: Optional[model.SessionStatus] = None,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[LongRunningSession], int]:
    """List sessions for a user."""
    db = await get_db()

    where_clause: dict[str, Any] = {"userId": user_id}
    if status:
        where_clause["status"] = status.value.upper()

    total = await db.longrunningsession.count(where=where_clause)

    sessions = await db.longrunningsession.find_many(
        where=where_clause,
        order_by={"updatedAt": "desc"},
        skip=(page - 1) * page_size,
        take=page_size,
    )

    return sessions, total


async def update_session(
    session_id: str,
    user_id: str,
    status: Optional[model.SessionStatus] = None,
    current_session_id: Optional[str] = None,
    current_feature_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Optional[LongRunningSession]:
    """Update a session."""
    db = await get_db()

    # Verify ownership
    existing = await get_session(session_id, user_id)
    if not existing:
        return None

    update_data: dict[str, Any] = {}
    if status is not None:
        update_data["status"] = status.value.upper()
    if current_session_id is not None:
        update_data["currentSessionId"] = current_session_id
    if current_feature_id is not None:
        update_data["currentFeatureId"] = current_feature_id
    if metadata is not None:
        update_data["metadata"] = metadata

    if not update_data:
        return existing

    # Increment session count when status changes to WORKING
    if status == model.SessionStatus.WORKING:
        update_data["sessionCount"] = existing.sessionCount + 1

    return await db.longrunningsession.update(
        where={"id": session_id},
        data=update_data,
    )


async def delete_session(session_id: str, user_id: str) -> bool:
    """Delete a session and all related data."""
    db = await get_db()

    # Verify ownership
    existing = await get_session(session_id, user_id)
    if not existing:
        return False

    await db.longrunningsession.delete(where={"id": session_id})
    return True


# === Feature Operations ===


async def create_feature(
    session_id: str,
    user_id: str,
    feature_id: str,
    description: str,
    category: model.FeatureCategory = model.FeatureCategory.FUNCTIONAL,
    steps: list[str] = [],
    priority: int = 5,
    dependencies: list[str] = [],
) -> Optional[LongRunningFeature]:
    """Create a new feature for a session."""
    db = await get_db()

    # Verify session ownership
    session = await get_session(session_id, user_id)
    if not session:
        return None

    return await db.longrunningfeature.create(
        data={
            "sessionId": session_id,
            "featureId": feature_id,
            "category": category.value.upper(),
            "description": description,
            "steps": steps,
            "priority": priority,
            "dependencies": dependencies,
            "status": "PENDING",
        }
    )


async def get_features(
    session_id: str,
    user_id: str,
    status: Optional[model.FeatureStatus] = None,
) -> list[LongRunningFeature]:
    """Get all features for a session."""
    db = await get_db()

    # Verify session ownership
    session = await get_session(session_id, user_id)
    if not session:
        return []

    where_clause: dict[str, Any] = {"sessionId": session_id}
    if status:
        where_clause["status"] = status.value.upper()

    return await db.longrunningfeature.find_many(
        where=where_clause,
        order_by=[{"priority": "asc"}, {"createdAt": "asc"}],
    )


async def update_feature(
    session_id: str,
    feature_id: str,
    user_id: str,
    status: Optional[model.FeatureStatus] = None,
    notes: Optional[str] = None,
    updated_by_session: Optional[str] = None,
) -> Optional[LongRunningFeature]:
    """Update a feature's status."""
    db = await get_db()

    # Verify session ownership
    session = await get_session(session_id, user_id)
    if not session:
        return None

    # Find the feature
    feature = await db.longrunningfeature.find_first(
        where={
            "sessionId": session_id,
            "featureId": feature_id,
        }
    )
    if not feature:
        return None

    update_data: dict[str, Any] = {}
    if status is not None:
        update_data["status"] = status.value.upper()
    if notes is not None:
        update_data["notes"] = notes
    if updated_by_session is not None:
        update_data["updatedBySession"] = updated_by_session

    if not update_data:
        return feature

    return await db.longrunningfeature.update(
        where={"id": feature.id},
        data=update_data,
    )


async def get_next_feature(
    session_id: str,
    user_id: str,
) -> Optional[LongRunningFeature]:
    """Get the next feature to work on (highest priority pending/failing)."""
    db = await get_db()

    # Verify session ownership
    session = await get_session(session_id, user_id)
    if not session:
        return None

    # Get failing features first (highest priority)
    failing = await db.longrunningfeature.find_first(
        where={
            "sessionId": session_id,
            "status": "FAILING",
        },
        order_by=[{"priority": "asc"}, {"createdAt": "asc"}],
    )
    if failing:
        return failing

    # Then get pending features
    return await db.longrunningfeature.find_first(
        where={
            "sessionId": session_id,
            "status": "PENDING",
        },
        order_by=[{"priority": "asc"}, {"createdAt": "asc"}],
    )


async def get_feature_summary(
    session_id: str,
    user_id: str,
) -> dict[str, int]:
    """Get a summary of feature statuses."""
    db = await get_db()

    # Verify session ownership
    session = await get_session(session_id, user_id)
    if not session:
        return {}

    features = await db.longrunningfeature.find_many(
        where={"sessionId": session_id}
    )

    summary: dict[str, int] = {
        "pending": 0,
        "in_progress": 0,
        "passing": 0,
        "failing": 0,
        "blocked": 0,
        "skipped": 0,
    }

    for f in features:
        status = f.status.lower()
        if status in summary:
            summary[status] += 1

    return summary


# === Progress Operations ===


async def create_progress_entry(
    session_id: str,
    user_id: str,
    agent_session_id: str,
    entry_type: model.ProgressEntryType,
    title: str,
    description: Optional[str] = None,
    feature_id: Optional[str] = None,
    git_commit_hash: Optional[str] = None,
    files_changed: list[str] = [],
    metadata: dict[str, Any] = {},
) -> Optional[LongRunningProgress]:
    """Create a progress entry."""
    db = await get_db()

    # Verify session ownership
    session = await get_session(session_id, user_id)
    if not session:
        return None

    # Find feature if feature_id is provided
    db_feature_id = None
    if feature_id:
        feature = await db.longrunningfeature.find_first(
            where={
                "sessionId": session_id,
                "featureId": feature_id,
            }
        )
        if feature:
            db_feature_id = feature.id

    return await db.longrunningprogress.create(
        data={
            "sessionId": session_id,
            "agentSessionId": agent_session_id,
            "entryType": entry_type.value.upper(),
            "title": title,
            "description": description,
            "featureId": db_feature_id,
            "gitCommitHash": git_commit_hash,
            "filesChanged": files_changed,
            "metadata": metadata,
        }
    )


async def get_progress_entries(
    session_id: str,
    user_id: str,
    limit: int = 50,
    agent_session_id: Optional[str] = None,
    feature_id: Optional[str] = None,
) -> list[LongRunningProgress]:
    """Get progress entries for a session."""
    db = await get_db()

    # Verify session ownership
    session = await get_session(session_id, user_id)
    if not session:
        return []

    where_clause: dict[str, Any] = {"sessionId": session_id}

    if agent_session_id:
        where_clause["agentSessionId"] = agent_session_id

    if feature_id:
        feature = await db.longrunningfeature.find_first(
            where={
                "sessionId": session_id,
                "featureId": feature_id,
            }
        )
        if feature:
            where_clause["featureId"] = feature.id

    return await db.longrunningprogress.find_many(
        where=where_clause,
        order_by={"createdAt": "desc"},
        take=limit,
    )
