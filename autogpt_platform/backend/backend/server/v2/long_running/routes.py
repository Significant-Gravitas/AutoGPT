"""
API routes for long-running agent session management.

Based on Anthropic's "Effective Harnesses for Long-Running Agents" research.
https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
"""

import logging
from typing import Optional

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, HTTPException, Query, Security, status

from . import db, model

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/long-running",
    tags=["long-running", "private"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)


# === Session Routes ===


@router.post(
    "/sessions",
    summary="Create Long-Running Session",
    response_model=model.SessionResponse,
    responses={
        201: {"description": "Session created successfully"},
        500: {"description": "Server error"},
    },
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    request: model.CreateSessionRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> model.SessionResponse:
    """
    Create a new long-running agent session.

    This is typically called by the initializer agent when starting
    a new multi-session project.
    """
    try:
        session = await db.create_session(
            user_id=user_id,
            project_name=request.project_name,
            project_description=request.project_description,
            working_directory=request.working_directory,
            features=request.features,
        )

        return model.SessionResponse(
            id=session.id,
            project_name=session.projectName,
            project_description=session.projectDescription,
            status=model.SessionStatus(session.status.lower()),
            current_session_id=session.currentSessionId,
            session_count=session.sessionCount,
            working_directory=session.workingDirectory,
            feature_list_path=session.featureListPath,
            progress_log_path=session.progressLogPath,
            init_script_path=session.initScriptPath,
            git_repo_initialized=session.gitRepoInitialized,
            current_feature_id=session.currentFeatureId,
            environment_variables=session.environmentVariables or {},
            metadata=session.metadata or {},
            created_at=session.createdAt,
            updated_at=session.updatedAt,
        )
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.get(
    "/sessions",
    summary="List Long-Running Sessions",
    response_model=model.SessionListResponse,
    responses={
        200: {"description": "List of sessions"},
        500: {"description": "Server error"},
    },
)
async def list_sessions(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
    session_status: Optional[model.SessionStatus] = Query(
        None, alias="status", description="Filter by session status"
    ),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
) -> model.SessionListResponse:
    """List all long-running sessions for the user."""
    try:
        sessions, total = await db.list_sessions(
            user_id=user_id,
            status=session_status,
            page=page,
            page_size=page_size,
        )

        return model.SessionListResponse(
            sessions=[
                model.SessionResponse(
                    id=s.id,
                    project_name=s.projectName,
                    project_description=s.projectDescription,
                    status=model.SessionStatus(s.status.lower()),
                    current_session_id=s.currentSessionId,
                    session_count=s.sessionCount,
                    working_directory=s.workingDirectory,
                    feature_list_path=s.featureListPath,
                    progress_log_path=s.progressLogPath,
                    init_script_path=s.initScriptPath,
                    git_repo_initialized=s.gitRepoInitialized,
                    current_feature_id=s.currentFeatureId,
                    environment_variables=s.environmentVariables or {},
                    metadata=s.metadata or {},
                    created_at=s.createdAt,
                    updated_at=s.updatedAt,
                )
                for s in sessions
            ],
            total=total,
            page=page,
            page_size=page_size,
        )
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.get(
    "/sessions/{session_id}",
    summary="Get Long-Running Session",
    response_model=model.SessionDetailResponse,
    responses={
        200: {"description": "Session details"},
        404: {"description": "Session not found"},
        500: {"description": "Server error"},
    },
)
async def get_session(
    session_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> model.SessionDetailResponse:
    """Get detailed information about a long-running session."""
    try:
        session = await db.get_session_with_details(session_id, user_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )

        # Get feature summary
        summary = await db.get_feature_summary(session_id, user_id)

        features = [
            model.FeatureResponse(
                id=f.id,
                feature_id=f.featureId,
                category=model.FeatureCategory(f.category.lower()),
                description=f.description,
                steps=f.steps or [],
                status=model.FeatureStatus(f.status.lower()),
                priority=f.priority,
                dependencies=f.dependencies or [],
                notes=f.notes,
                updated_by_session=f.updatedBySession,
                created_at=f.createdAt,
                updated_at=f.updatedAt,
            )
            for f in (session.Features or [])
        ]

        progress = [
            model.ProgressEntryResponse(
                id=p.id,
                agent_session_id=p.agentSessionId,
                entry_type=model.ProgressEntryType(p.entryType.lower()),
                title=p.title,
                description=p.description,
                feature_id=p.featureId,
                git_commit_hash=p.gitCommitHash,
                files_changed=p.filesChanged or [],
                metadata=p.metadata or {},
                created_at=p.createdAt,
            )
            for p in (session.ProgressLog or [])
        ]

        return model.SessionDetailResponse(
            id=session.id,
            project_name=session.projectName,
            project_description=session.projectDescription,
            status=model.SessionStatus(session.status.lower()),
            current_session_id=session.currentSessionId,
            session_count=session.sessionCount,
            working_directory=session.workingDirectory,
            feature_list_path=session.featureListPath,
            progress_log_path=session.progressLogPath,
            init_script_path=session.initScriptPath,
            git_repo_initialized=session.gitRepoInitialized,
            current_feature_id=session.currentFeatureId,
            environment_variables=session.environmentVariables or {},
            metadata=session.metadata or {},
            created_at=session.createdAt,
            updated_at=session.updatedAt,
            features=features,
            recent_progress=progress,
            feature_summary=summary,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.patch(
    "/sessions/{session_id}",
    summary="Update Long-Running Session",
    response_model=model.SessionResponse,
    responses={
        200: {"description": "Session updated"},
        404: {"description": "Session not found"},
        500: {"description": "Server error"},
    },
)
async def update_session(
    session_id: str,
    request: model.UpdateSessionRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> model.SessionResponse:
    """Update a long-running session's status or metadata."""
    try:
        session = await db.update_session(
            session_id=session_id,
            user_id=user_id,
            status=request.status,
            current_session_id=request.current_session_id,
            current_feature_id=request.current_feature_id,
            metadata=request.metadata,
        )

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )

        return model.SessionResponse(
            id=session.id,
            project_name=session.projectName,
            project_description=session.projectDescription,
            status=model.SessionStatus(session.status.lower()),
            current_session_id=session.currentSessionId,
            session_count=session.sessionCount,
            working_directory=session.workingDirectory,
            feature_list_path=session.featureListPath,
            progress_log_path=session.progressLogPath,
            init_script_path=session.initScriptPath,
            git_repo_initialized=session.gitRepoInitialized,
            current_feature_id=session.currentFeatureId,
            environment_variables=session.environmentVariables or {},
            metadata=session.metadata or {},
            created_at=session.createdAt,
            updated_at=session.updatedAt,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.delete(
    "/sessions/{session_id}",
    summary="Delete Long-Running Session",
    responses={
        204: {"description": "Session deleted"},
        404: {"description": "Session not found"},
        500: {"description": "Server error"},
    },
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_session(
    session_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> None:
    """Delete a long-running session and all related data."""
    try:
        success = await db.delete_session(session_id, user_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


# === Feature Routes ===


@router.post(
    "/sessions/{session_id}/features",
    summary="Create Feature",
    response_model=model.FeatureResponse,
    responses={
        201: {"description": "Feature created"},
        404: {"description": "Session not found"},
        500: {"description": "Server error"},
    },
    status_code=status.HTTP_201_CREATED,
)
async def create_feature(
    session_id: str,
    request: model.CreateFeatureRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> model.FeatureResponse:
    """Create a new feature for a session."""
    try:
        feature = await db.create_feature(
            session_id=session_id,
            user_id=user_id,
            feature_id=request.feature_id,
            description=request.description,
            category=request.category,
            steps=request.steps,
            priority=request.priority,
            dependencies=request.dependencies,
        )

        if not feature:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )

        return model.FeatureResponse(
            id=feature.id,
            feature_id=feature.featureId,
            category=model.FeatureCategory(feature.category.lower()),
            description=feature.description,
            steps=feature.steps or [],
            status=model.FeatureStatus(feature.status.lower()),
            priority=feature.priority,
            dependencies=feature.dependencies or [],
            notes=feature.notes,
            updated_by_session=feature.updatedBySession,
            created_at=feature.createdAt,
            updated_at=feature.updatedAt,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create feature: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.get(
    "/sessions/{session_id}/features",
    summary="List Features",
    response_model=model.FeatureListResponse,
    responses={
        200: {"description": "List of features"},
        404: {"description": "Session not found"},
        500: {"description": "Server error"},
    },
)
async def list_features(
    session_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
    feature_status: Optional[model.FeatureStatus] = Query(
        None, alias="status", description="Filter by feature status"
    ),
) -> model.FeatureListResponse:
    """List all features for a session."""
    try:
        features = await db.get_features(session_id, user_id, feature_status)
        summary = await db.get_feature_summary(session_id, user_id)

        return model.FeatureListResponse(
            features=[
                model.FeatureResponse(
                    id=f.id,
                    feature_id=f.featureId,
                    category=model.FeatureCategory(f.category.lower()),
                    description=f.description,
                    steps=f.steps or [],
                    status=model.FeatureStatus(f.status.lower()),
                    priority=f.priority,
                    dependencies=f.dependencies or [],
                    notes=f.notes,
                    updated_by_session=f.updatedBySession,
                    created_at=f.createdAt,
                    updated_at=f.updatedAt,
                )
                for f in features
            ],
            total=len(features),
            summary=summary,
        )
    except Exception as e:
        logger.error(f"Failed to list features: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.get(
    "/sessions/{session_id}/features/next",
    summary="Get Next Feature",
    response_model=Optional[model.FeatureResponse],
    responses={
        200: {"description": "Next feature to work on"},
        404: {"description": "Session not found or no features available"},
        500: {"description": "Server error"},
    },
)
async def get_next_feature(
    session_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Optional[model.FeatureResponse]:
    """Get the next feature to work on (highest priority pending/failing)."""
    try:
        feature = await db.get_next_feature(session_id, user_id)

        if not feature:
            return None

        return model.FeatureResponse(
            id=feature.id,
            feature_id=feature.featureId,
            category=model.FeatureCategory(feature.category.lower()),
            description=feature.description,
            steps=feature.steps or [],
            status=model.FeatureStatus(feature.status.lower()),
            priority=feature.priority,
            dependencies=feature.dependencies or [],
            notes=feature.notes,
            updated_by_session=feature.updatedBySession,
            created_at=feature.createdAt,
            updated_at=feature.updatedAt,
        )
    except Exception as e:
        logger.error(f"Failed to get next feature: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.patch(
    "/sessions/{session_id}/features/{feature_id}",
    summary="Update Feature",
    response_model=model.FeatureResponse,
    responses={
        200: {"description": "Feature updated"},
        404: {"description": "Feature not found"},
        500: {"description": "Server error"},
    },
)
async def update_feature(
    session_id: str,
    feature_id: str,
    request: model.UpdateFeatureRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> model.FeatureResponse:
    """Update a feature's status."""
    try:
        feature = await db.update_feature(
            session_id=session_id,
            feature_id=feature_id,
            user_id=user_id,
            status=request.status,
            notes=request.notes,
            updated_by_session=request.updated_by_session,
        )

        if not feature:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Feature {feature_id} not found in session {session_id}",
            )

        return model.FeatureResponse(
            id=feature.id,
            feature_id=feature.featureId,
            category=model.FeatureCategory(feature.category.lower()),
            description=feature.description,
            steps=feature.steps or [],
            status=model.FeatureStatus(feature.status.lower()),
            priority=feature.priority,
            dependencies=feature.dependencies or [],
            notes=feature.notes,
            updated_by_session=feature.updatedBySession,
            created_at=feature.createdAt,
            updated_at=feature.updatedAt,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update feature: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


# === Progress Routes ===


@router.post(
    "/sessions/{session_id}/progress",
    summary="Create Progress Entry",
    response_model=model.ProgressEntryResponse,
    responses={
        201: {"description": "Progress entry created"},
        404: {"description": "Session not found"},
        500: {"description": "Server error"},
    },
    status_code=status.HTTP_201_CREATED,
)
async def create_progress_entry(
    session_id: str,
    request: model.CreateProgressEntryRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> model.ProgressEntryResponse:
    """Create a progress entry for a session."""
    try:
        entry = await db.create_progress_entry(
            session_id=session_id,
            user_id=user_id,
            agent_session_id=request.agent_session_id,
            entry_type=request.entry_type,
            title=request.title,
            description=request.description,
            feature_id=request.feature_id,
            git_commit_hash=request.git_commit_hash,
            files_changed=request.files_changed,
            metadata=request.metadata,
        )

        if not entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )

        return model.ProgressEntryResponse(
            id=entry.id,
            agent_session_id=entry.agentSessionId,
            entry_type=model.ProgressEntryType(entry.entryType.lower()),
            title=entry.title,
            description=entry.description,
            feature_id=entry.featureId,
            git_commit_hash=entry.gitCommitHash,
            files_changed=entry.filesChanged or [],
            metadata=entry.metadata or {},
            created_at=entry.createdAt,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create progress entry: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.get(
    "/sessions/{session_id}/progress",
    summary="Get Progress Entries",
    response_model=list[model.ProgressEntryResponse],
    responses={
        200: {"description": "List of progress entries"},
        404: {"description": "Session not found"},
        500: {"description": "Server error"},
    },
)
async def get_progress_entries(
    session_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
    limit: int = Query(50, ge=1, le=200, description="Maximum entries to return"),
    agent_session_id: Optional[str] = Query(
        None, description="Filter by agent session ID"
    ),
    feature_id: Optional[str] = Query(None, description="Filter by feature ID"),
) -> list[model.ProgressEntryResponse]:
    """Get progress entries for a session."""
    try:
        entries = await db.get_progress_entries(
            session_id=session_id,
            user_id=user_id,
            limit=limit,
            agent_session_id=agent_session_id,
            feature_id=feature_id,
        )

        return [
            model.ProgressEntryResponse(
                id=e.id,
                agent_session_id=e.agentSessionId,
                entry_type=model.ProgressEntryType(e.entryType.lower()),
                title=e.title,
                description=e.description,
                feature_id=e.featureId,
                git_commit_hash=e.gitCommitHash,
                files_changed=e.filesChanged or [],
                metadata=e.metadata or {},
                created_at=e.createdAt,
            )
            for e in entries
        ]
    except Exception as e:
        logger.error(f"Failed to get progress entries: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
