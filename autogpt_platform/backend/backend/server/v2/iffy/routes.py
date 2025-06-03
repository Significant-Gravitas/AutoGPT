import logging

from autogpt_libs.auth.middleware import HMACValidator
from autogpt_libs.utils.cache import thread_cached
from fastapi import APIRouter, Depends, HTTPException, Request, Response

from backend.util.service import get_service_client
from backend.util.settings import Settings
from backend.server.routers.v1 import _cancel_execution

from .models import BlockContentForModeration, EventType, IffyWebhookEvent, UserData

logger = logging.getLogger(__name__)
settings = Settings()

iffy_router = APIRouter()

iffy_signature_validator = HMACValidator(
    header_name="X-Signature",
    secret=settings.secrets.iffy_webhook_secret,
    error_message="Invalid Iffy signature",
)


# This handles the webhook events from iffy like stopping an execution if a flagged block is detected.
async def handle_record_event(
    event_type: EventType, block_content: BlockContentForModeration
) -> Response:
    """Handle record-related webhook events
    If any blocks are flagged, we stop the execution and log the event."""

    if event_type == EventType.RECORD_FLAGGED:
        logger.warning(
            f'Content flagged for node "{block_content.node_id}" ("{block_content.block_name}") '
            f'in execution "{block_content.graph_exec_id}"'
        )
        
        # Stop execution directly
        await _cancel_execution(block_content.graph_exec_id)
        logger.info(f'Successfully stopped execution "{block_content.graph_exec_id}" due to flagged content')

        return Response(status_code=200)

    elif event_type in (EventType.RECORD_COMPLIANT, EventType.RECORD_UNFLAGGED):
        logger.info(
            f'Content cleared for node "{block_content.node_id}" ("{block_content.block_name}") '
            f'in execution "{block_content.graph_exec_id}"'
        )

    return Response(status_code=200)


async def handle_user_event(event_type: EventType, user_payload: UserData) -> Response:
    """Handle user-related webhook events
    For now we are just logging these events from iffy
    and replying with a 200 status code to keep iffy happy and to prevent it from retrying the request.
    """

    user_id = user_payload.clientId
    if not user_id:
        logger.error("Received user event without user ID")
        raise HTTPException(
            status_code=400,
            detail="Missing required field 'clientId' in user event payload",
        )

    status_updated_at = user_payload.get("statusUpdatedAt", "unknown time")
    status_updated_via = user_payload.get("statusUpdatedVia", "unknown method")

    event_messages = {
        EventType.USER_SUSPENDED: f'User "{user_id}" has been SUSPENDED via {status_updated_via} at {status_updated_at}',
        EventType.USER_UNSUSPENDED: f'User "{user_id}" has been UNSUSPENDED via {status_updated_via} at {status_updated_at}',
        EventType.USER_COMPLIANT: f'User "{user_id}" has been marked as COMPLIANT via {status_updated_via} at {status_updated_at}',
        # Users can only be manually banned and unbanned on the iffy dashboard, for now logging these events
        EventType.USER_BANNED: f'User "{user_id}" has been BANNED via {status_updated_via} at {status_updated_at}',
        EventType.USER_UNBANNED: f'User "{user_id}" has been UNBANNED via {status_updated_via} at {status_updated_at}',
    }

    if event_type in event_messages:
        log_message = event_messages[event_type]
        (
            logger.warning(log_message)
            if "suspended" in event_type or "banned" in event_type
            else logger.info(log_message)
        )

    return Response(status_code=200)


@iffy_router.post("/webhook")
async def handle_iffy_webhook(
    request: Request, _=Depends(iffy_signature_validator.get_dependency())
) -> Response:
    body = await request.body()
    try:
        event_data = IffyWebhookEvent.model_validate_json(body)
    except Exception as e:
        logger.error(f"Failed to parse Iffy webhook data: {e}")
        raise HTTPException(status_code=400, detail="Invalid request body")

    try:
        if event_data.event.startswith("record."):
            metadata = event_data.payload.get("metadata", {})
            block_content = BlockContentForModeration(
                graph_id=metadata.get("graphId", ""),
                graph_exec_id=metadata.get("graphExecutionId", ""),
                node_id=metadata.get("nodeId", ""),
                block_id=metadata.get("blockId", ""),
                block_name=metadata.get("blockName", "Unknown Block"),
                block_type=metadata.get("blockType", ""),
                input_data=metadata.get("inputData", {}),
            )
            return await handle_record_event(event_data.event, block_content)
        elif event_data.event.startswith("user."):
            # Create UserData from payload
            user_data = UserData(**event_data.payload)
            return await handle_user_event(event_data.event, user_data)
        else:
            logger.info(f"Received unhandled Iffy event: {event_data.event}")
            return Response(status_code=200)
    except Exception as e:
        if "not active/running" in str(e):
            return Response(status_code=200)
        raise HTTPException(status_code=200, detail=str(e))
