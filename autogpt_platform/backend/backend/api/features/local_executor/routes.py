"""Authenticated HTTP routes for the Local PC executor."""

from __future__ import annotations

import logging
from typing import Annotated, NoReturn

from autogpt_libs import auth
from fastapi import APIRouter, Depends, HTTPException, Security, status
from pydantic import ValidationError

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
from backend.api.features.local_executor.state import mark_recording_reviewed
from backend.api.features.local_executor.websocket import router as websocket_router
from backend.copilot.config import ChatConfig
from backend.copilot.model import (
    CHAT_STATUS_RUNNING,
    ChatSessionInfo,
    get_chat_session_metadata,
)
from backend.copilot.rate_limit import enforce_payment_paywall
from backend.copilot.sdk.computer_use_policy import computer_use_scope
from backend.copilot.sdk.recording_tools import (
    RecordingStateUnavailable,
    RecordingSummaryUnavailable,
    register_recording_reviewed,
    start_recording,
    stop_recording,
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
from backend.copilot.tools.recording_models import RecordingReviewApplied

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
    dependencies=[Depends(enforce_payment_paywall)],
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
    except ValidationError as exc:
        # The listing is the companion's data; never echo it back.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The Local PC executor returned an invalid folder listing",
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
        driver=_computer_use_driver_label(hello),
        tools=_grantable_computer_use_tools(hello),
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
        computer_use_driver=_computer_use_driver_label(hello),
        computer_use_tools=_grantable_computer_use_tools(hello),
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
    if session.metadata.execution_target.kind != "local":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This session is not bound to a Local PC executor",
        )
    await _require_local_executor_enabled(user_id)
    hello = await _get_session_executor_hello(session, user_id)
    if request.approved:
        if hello is None or not hello.machine_id or computer_use_scope(hello) is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="No computer-use capable Local PC executor is connected",
            )
        if (
            request.expected_machine_id != hello.machine_id
            or request.expected_driver != _computer_use_driver_label(hello)
            or request.expected_tools != _grantable_computer_use_tools(hello)
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
        driver=_computer_use_driver_label(hello),
        tools=_grantable_computer_use_tools(hello),
    )
    return ComputerUseConsentResponse(computer_use_consent=consent)


@router.post(
    "/api/copilot/sessions/{session_id}/executor/recording/start",
    tags=["copilot"],
    dependencies=[Depends(enforce_payment_paywall)],
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
        recording_id = await start_recording(
            shim,
            session_id,
            mode=request.mode,
            interpretation_route=request.interpretation_route,
            channels=list(request.channels),
        )
    except ShimRecordingError as exc:
        _raise_recording_http_error(exc)
    except RecordingStateUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": exc.code,
                "message": exc.message,
                "details": {"recording_id": exc.recording_id},
            },
        ) from exc
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
        # The non-idempotent STOP is saved before the idempotent FETCH, so a
        # retry after a failed FETCH or response recovers without a second.
        summary = await stop_recording(shim, session_id, request.recording_id)
        recording = await shim.recording.fetch(request.recording_id)
    except ShimRecordingError as exc:
        _raise_recording_http_error(exc)
    except RecordingSummaryUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
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
    if target.kind != "local":
        return None
    hello = await get_shim_manager().get_hello_async(session.session_id)
    if hello is not None:
        if (
            hello.machine_id == target.machine_id
            and hello.allowed_root == target.allowed_root
        ):
            return hello
        logger.warning(
            "[LocalPC] Ignoring mismatched data HELLO for session %s",
            session.session_id[:12],
        )

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
    """Detach the reviewed session from its machine and close its shim, as a
    turn does when it ends, unless a turn is running: it shares this shim,
    and releases it itself when it ends."""
    try:
        session = await get_chat_session_metadata(session_id, user_id)
    except Exception:
        logger.warning(
            "[LocalPC] Failed to read reviewed recording session %s",
            session_id[:12],
            exc_info=True,
        )
        session = None
    if session is not None and session.chat_status == CHAT_STATUS_RUNNING:
        return
    try:
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
    target = session.metadata.execution_target
    if target.kind != "local":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This session is not bound to a Local PC executor",
        )
    await _require_local_executor_enabled(user_id)
    if recording:
        await _require_workflow_recording_enabled(user_id)
    try:
        shim = await get_shim_manager().get_or_create_shim_for_session(
            session_id, timeout=1.0
        )
        if (
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


def _grantable_computer_use_tools(hello: ShimHello | None) -> list[str] | None:
    """The driver tools approving computer use would grant: the same scope
    the computer-use tools check at each call."""
    scope = computer_use_scope(hello) if hello is not None else None
    return list(scope.tools) if scope is not None else None


def _computer_use_driver_label(hello: ShimHello | None) -> str | None:
    scope = computer_use_scope(hello) if hello is not None else None
    return scope.driver if scope is not None else None


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
