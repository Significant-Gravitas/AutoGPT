"""Authenticated HTTP routes for the Local PC executor."""

from __future__ import annotations

import logging
from typing import Annotated, NoReturn

from autogpt_libs import auth
from fastapi import APIRouter, HTTPException, Security, status

from backend.api.features.local_executor.consent import (
    get_computer_use_consent,
    set_computer_use_consent,
)
from backend.api.features.local_executor.gating import (
    is_local_executor_enabled,
    is_workflow_recording_enabled,
)
from backend.api.features.local_executor.models import (
    ComputerUseConsentRequest,
    ComputerUseConsentResponse,
    DirectoryListRequest,
    DirectoryListResponse,
    ExecutorMachine,
    ExecutorsResponse,
    ExecutorStatus,
    MachineID,
    RecordingID,
    RecordingReviewRequest,
    RecordingStartRequest,
    RecordingStartResponse,
    RecordingStopRequest,
    RecordingStopResponse,
    SessionID,
)
from backend.api.features.local_executor.state import (
    get_recording_state,
    mark_recording_reviewed,
    mark_recording_started,
    mark_recording_stopped,
)
from backend.api.features.local_executor.websocket import router as websocket_router
from backend.copilot.config import ChatConfig
from backend.copilot.model import ChatSessionInfo, get_chat_session_metadata
from backend.copilot.sdk.recording_tools import (
    register_recording_reviewed,
    register_recording_started,
    register_recording_stopped,
)
from backend.copilot.tools.local_pc_machine import (
    MachineConnectionStaleError,
    MachineControlError,
    MachineNotConnectedError,
    MachineSessionBinding,
    detach_machine_session,
    get_machine_presence,
    list_machine_presences,
    machine_rpc,
    restore_machine_session,
)
from backend.copilot.tools.local_pc_shim import (
    LocalPCShim,
    ShimHello,
    ShimRecordingError,
    get_shim_manager,
)
from backend.copilot.tools.recording_models import (
    RecordingReviewApplied,
    RecordingSummary,
)

router = APIRouter()
router.include_router(websocket_router)
logger = logging.getLogger(__name__)


@router.get(
    "/api/copilot/executors",
    response_model=ExecutorsResponse,
    tags=["copilot"],
)
async def list_local_executors(
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> ExecutorsResponse:
    """List this user's currently connected persistent Local PC executors."""
    await _require_local_executor_enabled(user_id)
    client_id = ChatConfig().local_pc_executor_oauth_client_id
    presences = await list_machine_presences(user_id, client_id)
    return ExecutorsResponse(
        executors=[
            ExecutorMachine(
                machine_id=str(presence.hello["machine_id"]),
                connection_id=presence.connection_id,
                display_name=str(
                    presence.hello.get("display_name") or presence.hello["machine_id"]
                ),
                platform=str(presence.hello.get("platform") or ""),
                arch=str(presence.hello.get("arch") or ""),
                shim_version=str(presence.hello.get("shim_version") or ""),
                capabilities=[
                    str(capability)
                    for capability in presence.hello.get("capabilities") or []
                    if isinstance(capability, str)
                ],
            )
            for presence in presences
        ]
    )


@router.post(
    "/api/copilot/executors/{machine_id}/directories",
    response_model=DirectoryListResponse,
    tags=["copilot"],
)
async def list_local_executor_directories(
    machine_id: MachineID,
    request: DirectoryListRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> DirectoryListResponse:
    """Browse directories through an owner-scoped persistent machine channel."""
    await _require_local_executor_enabled(user_id)
    client_id = ChatConfig().local_pc_executor_oauth_client_id
    try:
        presence = await get_machine_presence(
            user_id,
            client_id,
            machine_id,
            expected_connection_id=request.expected_connection_id,
        )
        message = await machine_rpc(
            presence,
            "DIRECTORY_LIST_REQUEST",
            request.model_dump(exclude={"expected_connection_id"}, exclude_none=True),
        )
        payload = message["payload"]
        return DirectoryListResponse.model_validate(
            {**payload, "connection_id": presence.connection_id}
        )
    except (MachineNotConnectedError, MachineConnectionStaleError) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except MachineControlError as exc:
        status_code = (
            status.HTTP_409_CONFLICT
            if exc.code in {"DIRECTORY_REFERENCE_INVALID", "DIRECTORY_UNAVAILABLE"}
            else status.HTTP_502_BAD_GATEWAY
        )
        raise HTTPException(
            status_code=status_code,
            detail={"code": exc.code, "message": str(exc), "details": exc.details},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc


@router.get(
    "/api/copilot/sessions/{session_id}/executor",
    response_model=ExecutorStatus,
    tags=["copilot"],
)
async def get_session_executor(
    session_id: SessionID,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> ExecutorStatus:
    """Return executor metadata only when the caller owns the session."""
    session = await _require_owned_session(session_id, user_id)
    hello = await _get_session_executor_hello(session, user_id)
    consent = await get_computer_use_consent(
        session_id,
        user_id,
        machine_id=hello.machine_id if hello else None,
        features_coarse=hello.computer_use_features_coarse if hello else None,
        features=hello.computer_use_features if hello else None,
    )
    if hello is None:
        return ExecutorStatus(kind="none", computer_use_consent=consent)
    return ExecutorStatus(
        kind="shim",
        computer_use_consent=consent,
        platform=hello.platform or None,
        arch=hello.arch or None,
        allowed_root=hello.allowed_root or None,
        machine_id=hello.machine_id or None,
        shim_version=hello.shim_version or None,
        capabilities=hello.capabilities or None,
        computer_use_features=hello.computer_use_features or None,
        computer_use_features_coarse=hello.computer_use_features_coarse or None,
        recording_channels=hello.recording_channels or None,
        recording_routes=hello.recording_routes or None,
    )


@router.post(
    "/api/copilot/sessions/{session_id}/executor/consent",
    tags=["copilot"],
)
async def set_session_computer_use_consent(
    session_id: SessionID,
    request: ComputerUseConsentRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> ComputerUseConsentResponse:
    """Persist the authenticated owner's explicit per-session decision."""
    session = await _require_owned_session(session_id, user_id)
    await _require_local_executor_enabled(user_id)
    hello = await _get_session_executor_hello(session, user_id)
    if request.approved:
        if (
            hello is None
            or not hello.machine_id
            or "computer_use" not in hello.capabilities
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="No computer-use capable Local PC executor is connected",
            )
        if (
            request.expected_machine_id != hello.machine_id
            or request.expected_features_coarse
            != _normalized_computer_use_features_coarse(hello)
            or request.expected_features != _normalized_computer_use_features(hello)
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The Local PC executor scope changed; review it and try again",
            )
    consent = await set_computer_use_consent(
        session_id,
        user_id,
        approved=request.approved,
        machine_id=hello.machine_id if hello else None,
        features_coarse=hello.computer_use_features_coarse if hello else None,
        features=hello.computer_use_features if hello else None,
    )
    return ComputerUseConsentResponse(computer_use_consent=consent)


@router.post(
    "/api/copilot/sessions/{session_id}/executor/recording/start",
    tags=["copilot"],
)
async def start_session_recording(
    session_id: SessionID,
    request: RecordingStartRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> RecordingStartResponse:
    """Request native shim consent, then start recording without exposing its token."""
    shim = await _require_owned_shim(session_id, user_id, recording=True)
    _require_recording_capability(shim)
    try:
        recording_id = await shim.recording.start_with_consent(
            mode=request.mode,
            interpretation_route=request.interpretation_route,
            channels=list(request.channels),
        )
    except ShimRecordingError as exc:
        _raise_recording_http_error(exc)
    if not recording_id:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The Local PC executor did not return a recording ID",
        )
    effective_route = shim.recording.effective_route_for(
        recording_id, request.interpretation_route
    )
    try:
        await mark_recording_started(
            session_id,
            recording_id,
            mode=request.mode,
            interpretation_route=effective_route,
            channels=list(request.channels),
        )
    except Exception as exc:
        try:
            await shim.recording.stop(recording_id)
        except Exception:
            logger.exception(
                "Failed to stop recording %s after state persistence failed",
                recording_id,
            )
        finally:
            shim.close_recording(recording_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recording state could not be persisted; recording was stopped",
        ) from exc
    register_recording_started(
        shim,
        recording_id,
        mode=request.mode,
        interpretation_route=effective_route,
        channels=list(request.channels),
    )
    return RecordingStartResponse(recording_id=recording_id)


@router.post(
    "/api/copilot/sessions/{session_id}/executor/recording/stop",
    tags=["copilot"],
)
async def stop_session_recording(
    session_id: SessionID,
    request: RecordingStopRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> RecordingStopResponse:
    """Stop and fetch the redacted recording for user review."""
    # Cleanup remains available after the workflow-recording rollout switch is
    # disabled. Start/review stay gated; stop still requires the session owner,
    # Local PC rollout, a live shim, and the recording capability.
    shim = await _require_owned_shim(session_id, user_id)
    _require_recording_capability(shim)
    try:
        summary = await _get_stopped_recording_summary(session_id, request.recording_id)
        if summary is None:
            summary = await shim.recording.stop(request.recording_id)
            # Persist the successful, non-idempotent STOP before the idempotent
            # FETCH. If FETCH or the HTTP response fails, a retry can recover
            # without sending a second STOP to the shim.
            await mark_recording_stopped(
                session_id, request.recording_id, summary=summary.to_dict()
            )
            register_recording_stopped(shim, request.recording_id, summary)
        recording = await shim.recording.fetch(request.recording_id)
    except ShimRecordingError as exc:
        _raise_recording_http_error(exc)
    finally:
        shim.close_recording(request.recording_id)
    return RecordingStopResponse(summary=summary, recording=recording)


@router.post(
    "/api/copilot/sessions/{session_id}/executor/recording/{recording_id}/review",
    tags=["copilot"],
)
async def review_session_recording(
    session_id: SessionID,
    recording_id: RecordingID,
    request: RecordingReviewRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> RecordingReviewApplied:
    """Apply the owner's removals and redactions to shim-owned recording data."""
    shim = await _require_owned_shim(session_id, user_id, recording=True)
    _require_recording_capability(shim)
    try:
        applied = await shim.recording.apply_review(
            recording_id,
            removed_step_seqs=request.removed_step_seqs,
            redacted_step_seqs=request.redacted_step_seqs,
        )
    except ShimRecordingError as exc:
        _raise_recording_http_error(exc)
    await mark_recording_reviewed(
        session_id, recording_id, step_count=applied.step_count
    )
    register_recording_reviewed(shim, recording_id, applied)
    await _release_reviewed_local_shim(session_id, user_id, shim)
    return applied


async def _require_owned_session(session_id: str, user_id: str) -> ChatSessionInfo:
    session = await get_chat_session_metadata(session_id, user_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    return session


async def _get_session_executor_hello(
    session: ChatSessionInfo,
    user_id: str,
) -> ShimHello | None:
    """Resolve an active child HELLO or project the persistent machine HELLO."""
    target = session.metadata.execution_target
    hello = await get_shim_manager().get_hello_async(session.session_id)
    if hello is not None:
        if target.kind != "local" or (
            hello.machine_id == target.machine_id
            and hello.allowed_root == target.allowed_root
        ):
            return hello
        logger.warning(
            "[LocalPC] Ignoring mismatched data HELLO for session %s",
            session.session_id[:12],
        )

    if target.kind != "local":
        return None
    try:
        presence = await get_machine_presence(
            user_id,
            ChatConfig().local_pc_executor_oauth_client_id,
            target.machine_id,
        )
        return ShimHello.from_payload(
            {**presence.hello, "allowed_root": target.allowed_root}
        )
    except (MachineNotConnectedError, MachineConnectionStaleError, ValueError):
        return None


async def _restore_owned_local_shim(
    session: ChatSessionInfo,
    user_id: str,
) -> LocalPCShim:
    target = session.metadata.execution_target
    if target.kind != "local":
        raise ConnectionError("This session is not bound to a Local PC executor")
    presence = await get_machine_presence(
        user_id,
        ChatConfig().local_pc_executor_oauth_client_id,
        target.machine_id,
    )
    binding = MachineSessionBinding(
        session_id=session.session_id,
        allowed_root=target.allowed_root,
        fingerprint=target.root_fingerprint,
        revision=target.revision,
        root_grant=target.root_grant,
    )
    restored = await restore_machine_session(presence, binding)
    if restored != binding:
        raise ConnectionError("The Local PC executor restored a different binding")
    shim = await get_shim_manager().get_or_create_shim_for_session(
        session.session_id, timeout=8.0
    )
    if shim.machine_id != target.machine_id or shim.allowed_root != target.allowed_root:
        await shim.kill()
        raise ConnectionError(
            "The Local PC executor data channel does not match the session binding"
        )
    return shim


async def _release_reviewed_local_shim(
    session_id: str,
    user_id: str,
    shim: LocalPCShim,
) -> None:
    try:
        session = await get_chat_session_metadata(session_id, user_id)
        if session is not None and session.metadata.execution_target.kind == "local":
            target = session.metadata.execution_target
            presence = await get_machine_presence(
                user_id,
                ChatConfig().local_pc_executor_oauth_client_id,
                target.machine_id,
            )
            await detach_machine_session(presence, session_id)
    except Exception:
        logger.warning(
            "[LocalPC] Failed to detach reviewed recording session %s",
            session_id[:12],
            exc_info=True,
        )
    finally:
        await shim.kill()


async def _require_owned_shim(
    session_id: str, user_id: str, *, recording: bool = False
) -> LocalPCShim:
    session = await _require_owned_session(session_id, user_id)
    await _require_local_executor_enabled(user_id)
    if recording:
        await _require_workflow_recording_enabled(user_id)
    try:
        shim = await get_shim_manager().get_or_create_shim_for_session(
            session_id, timeout=1.0
        )
        target = session.metadata.execution_target
        if target.kind == "local" and (
            shim.machine_id != target.machine_id
            or shim.allowed_root != target.allowed_root
        ):
            await shim.kill()
            raise ConnectionError(
                "The connected Local PC executor does not match the session binding"
            )
        return shim
    except (ConnectionError, TimeoutError):
        try:
            return await _restore_owned_local_shim(session, user_id)
        except Exception as restore_exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="No Local PC executor is connected for this session",
            ) from restore_exc


async def _require_local_executor_enabled(user_id: str) -> None:
    if not await is_local_executor_enabled(user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Local PC executor is not enabled",
        )


async def _require_workflow_recording_enabled(user_id: str) -> None:
    if not await is_workflow_recording_enabled(user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow recording is not enabled",
        )


def _require_recording_capability(shim: LocalPCShim) -> None:
    if "recording" not in shim.capabilities:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The connected Local PC executor does not support recording",
        )


def _normalized_computer_use_features_coarse(hello: ShimHello) -> list[str]:
    return sorted(set(hello.computer_use_features_coarse))


def _normalized_computer_use_features(hello: ShimHello) -> list[str]:
    return sorted(set(hello.computer_use_features))


async def _get_stopped_recording_summary(
    session_id: str, recording_id: str
) -> RecordingSummary | None:
    state = await get_recording_state(session_id, recording_id)
    if state is None or state.status not in {"stopped", "reviewed"}:
        return None
    if state.summary is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stopped recording state is missing its recovery summary",
        )
    summary = RecordingSummary.from_payload(state.summary)
    if summary.recording_id != recording_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stopped recording state has an invalid recovery summary",
        )
    return summary


def _raise_recording_http_error(exc: ShimRecordingError) -> NoReturn:
    if exc.code in {"CONSENT_DENIED", "CONSENT_REQUIRED"}:
        status_code = status.HTTP_403_FORBIDDEN
    elif exc.code == "RECORDING_NOT_FOUND":
        status_code = status.HTTP_404_NOT_FOUND
    elif exc.code == "RECORDING_ALREADY_ACTIVE":
        status_code = status.HTTP_409_CONFLICT
    elif exc.code in {
        "RECORDING_CHANNEL_UNAVAILABLE",
        "INTERPRETATION_UNAVAILABLE",
        "CONSENT_SCOPE_MISMATCH",
        "RECORDING_REVIEW_INVALID",
    }:
        status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    else:
        status_code = status.HTTP_502_BAD_GATEWAY
    raise HTTPException(
        status_code=status_code,
        detail={"code": exc.code, "message": str(exc), "details": exc.details},
    ) from exc
