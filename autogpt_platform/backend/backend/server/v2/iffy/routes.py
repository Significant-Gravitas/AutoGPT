import hmac
import hashlib
import logging
from typing import Dict, Any
from fastapi import APIRouter, Request, Response, HTTPException, Header, Depends
from backend.util.settings import Settings
from backend.util.service import get_service_client
from backend.executor import ExecutionManager
from .models import EventType, IffyWebhookEvent
from autogpt_libs.auth.middleware import HMACValidator

logger = logging.getLogger(__name__)
settings = Settings()

iffy_router = APIRouter()

iffy_signature_validator = HMACValidator(
    header_name="X-Signature",
    secret=settings.secrets.iffy_webhook_secret,
    error_message="Invalid Iffy signature"
)

# This handles the webhook events from iffy like stopping an execution if a flagged block is detected.
async def handle_record_event(event_type: EventType, metadata: Dict[str, Any]) -> Response:
    """Handle record-related webhook events
        If any blocks are flagged, we stop the execution and log the event."""
    
    graph_exec_id = metadata.get("graphExecutionId")
    node_id = metadata.get("nodeId")
    block_name = metadata.get("blockName", "Unknown Block")

    if event_type == EventType.RECORD_FLAGGED:
        logger.warning(
            f'Content flagged for node "{node_id}" ("{block_name}") '
            f'in execution "{graph_exec_id}"'
        )
        execution_manager = get_service_client(ExecutionManager)
        try:
            execution_manager.cancel_execution(graph_exec_id)
            logger.info(f'Successfully stopped execution "{graph_exec_id}" due to flagged content')
        except Exception as e:
            if "not active/running" not in str(e):
                logger.error(f"Error cancelling execution processes: {str(e)}")
                raise
            logger.info(f'Execution "{graph_exec_id}" was already completed/cancelled')
        
        return Response(status_code=200)
    
    elif event_type in (EventType.RECORD_COMPLIANT, EventType.RECORD_UNFLAGGED):
        logger.info(
            f'Content cleared for node "{node_id}" ("{block_name}") '
            f'in execution "{graph_exec_id}"'
        )

    return Response(status_code=200)

async def handle_user_event(event_type: EventType, payload: Dict[str, Any]) -> Response:
    """Handle user-related webhook events
        For now we are just logging these events from iffy
        and replying with a 200 status code to keep iffy happy and to prevent it from retrying the request."""
    
    user_id = payload.get("clientId")
    if not user_id:
        logger.error("Received user event without user ID")
        raise HTTPException(
            status_code=400,
            detail="Missing required field 'clientId' in user event payload"
        )

    status_updated_at = payload.get("statusUpdatedAt")
    status_updated_via = payload.get("statusUpdatedVia")

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
        logger.warning(log_message) if "suspended" in event_type or "banned" in event_type else logger.info(log_message)

    return Response(status_code=200)

@iffy_router.post("/webhook")
async def handle_iffy_webhook(
    request: Request,
    _ = Depends(iffy_signature_validator.get_dependency())
) -> Response:
    body = await request.body()
    try:
        event_data = IffyWebhookEvent.model_validate_json(body)
    except Exception as e:
        logger.error(f"Failed to parse Iffy webhook data: {e}")
        raise HTTPException(status_code=400, detail="Invalid request body")

    try:
        if event_data.event.startswith("record."):
            return await handle_record_event(event_data.event, event_data.payload.get("metadata", {}))
        elif event_data.event.startswith("user."):
            return await handle_user_event(event_data.event, event_data.payload)
        else:
            logger.info(f"Received unhandled Iffy event: {event_data.event}")
            return Response(status_code=200)
    except Exception as e:
        if "not active/running" in str(e):
            return Response(status_code=200)
        raise HTTPException(status_code=200, detail=str(e))