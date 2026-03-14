"""WhatsApp Business Cloud API integration for the ABN Co-Navigator.

This module provides two FastAPI routes that are mounted onto the main app:

    GET  /whatsapp/webhook  — Meta webhook verification handshake
    POST /whatsapp/webhook  — Incoming message handler

How it works
------------
1. A user sends a WhatsApp message to the registered business phone number.
2. Meta's Cloud API forwards it as a JSON POST to /whatsapp/webhook.
3. We extract the sender's phone number and message text.
4. We route the message to the matching CoachingSession (or start a new one).
5. We call the WhatsApp Send Message API to reply.

Session lifecycle (mirrors the Telegram bot)
--------------------------------------------
- First message / keyword "hi" / "start"  → opens a new CoachingSession
- Free text                                → forwarded to active session
- "done" / "end"                          → extracts summary, saves, replies
- "cancel"                                → discards session without saving
- "help"                                  → shows command list

Required environment variables
-------------------------------
    WHATSAPP_APP_SECRET        — Facebook App Secret (for signature verification)
    WHATSAPP_ACCESS_TOKEN      — Permanent System User access token
    WHATSAPP_PHONE_NUMBER_ID   — Phone Number ID from Meta developer portal
    WHATSAPP_VERIFY_TOKEN      — Token you choose; must match webhook config in Meta portal

Optional
--------
    WHATSAPP_APP_ID            — Facebook App ID (informational; not used at runtime)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Any, Dict

import requests as http_requests
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import PlainTextResponse

from autogpt.coaching.config import coaching_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/whatsapp", tags=["WhatsApp"])

# phone_number (E.164 string) → CoachingSession
_wa_sessions: Dict[str, Any] = {}

# ── Helpers ───────────────────────────────────────────────────────────────────

GRAPH_API_VERSION = "v19.0"
GRAPH_API_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"

HELP_TEXT = (
    "👋 *ABN Co-Navigator — WhatsApp commands*\n\n"
    "• Type *start* or *hi* — begin a new coaching session\n"
    "• Send any message   — chat with the Navigator\n"
    "• Type *done* or *end* — close session and receive summary\n"
    "• Type *cancel*        — discard session without saving\n"
    "• Type *help*          — show this message"
)


def _send_whatsapp_text(to: str, body: str) -> None:
    """Send a plain-text WhatsApp message via the Cloud API."""
    token = coaching_config.whatsapp_access_token
    phone_number_id = coaching_config.whatsapp_phone_number_id

    if not token or not phone_number_id:
        logger.error("WhatsApp not configured: missing WHATSAPP_ACCESS_TOKEN or WHATSAPP_PHONE_NUMBER_ID")
        return

    url = f"{GRAPH_API_BASE}/{phone_number_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": body},
    }
    resp = http_requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=10,
    )
    if resp.status_code not in (200, 201):
        logger.error(
            "WhatsApp send failed to=%s status=%d body=%s",
            to,
            resp.status_code,
            resp.text[:200],
        )


def _verify_signature(body_bytes: bytes, signature_header: str) -> bool:
    """Return True if X-Hub-Signature-256 matches the payload HMAC."""
    secret = coaching_config.whatsapp_app_secret
    if not secret:
        # If secret is not configured, skip verification (development only)
        logger.warning("WHATSAPP_APP_SECRET not set — skipping signature verification")
        return True
    expected = "sha256=" + hmac.new(
        secret.encode(), body_bytes, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header or "")


def _format_summary(summary) -> str:
    """Render a SessionSummary as a readable WhatsApp message."""
    wl = summary.weekly_log
    lines = [
        "✅ *Session Summary*\n",
        f"*Focus goal:* {wl.focus_goal or '—'}",
        f"*Mood:* {wl.mood_indicator or '—'}",
        f"*Environmental changes:* {wl.environmental_changes or '—'}",
    ]
    if wl.key_results:
        lines.append("\n*Key Results:*")
        for kr in wl.key_results:
            lines.append(f"  • {kr.description} — {kr.status_pct}%")
    unresolved = [o.description for o in wl.obstacles if not o.resolved]
    if unresolved:
        lines.append("\n⚠ *Open obstacles:*")
        for obs in unresolved:
            lines.append(f"  - {obs}")
    lines += [
        f"\n*Alert:* {summary.alerts.level.value.upper()} — {summary.alerts.reason}",
        f"\n*Coach note:* {summary.summary_for_coach}",
        "\nSession saved. See you next week! 🚢",
    ]
    return "\n".join(lines)


# ── Session logic ─────────────────────────────────────────────────────────────

def _handle_start(phone: str, sender_name: str) -> str:
    from autogpt.coaching.session import CoachingSession
    from autogpt.coaching.storage import get_past_sessions, get_user_objectives

    if phone in _wa_sessions:
        return "You already have an active session. Send *done* to close it or keep chatting!"

    client_id = f"wa_{phone}"

    objectives = []
    past_sessions = []
    try:
        objectives = get_user_objectives(client_id)
        past_sessions = get_past_sessions(client_id, limit=3)
    except Exception as exc:
        logger.warning("Could not load context for wa_phone=%s: %s", phone, exc)

    session = CoachingSession(
        client_id=client_id,
        client_name=sender_name,
        objectives=objectives,
        past_sessions=past_sessions,
    )
    _wa_sessions[phone] = session

    try:
        return session.open()
    except Exception as exc:
        logger.error("session.open failed for wa_phone=%s: %s", phone, exc)
        del _wa_sessions[phone]
        return "Sorry, I couldn't start the session right now. Please try again."


def _handle_end(phone: str) -> str:
    from autogpt.coaching.storage import save_session

    session = _wa_sessions.get(phone)
    if session is None:
        return "No active session. Send *start* to begin a coaching session."

    try:
        summary = session.extract_summary()
        save_session(summary)
        result = _format_summary(summary)
    except Exception as exc:
        logger.error("Session end failed for wa_phone=%s: %s", phone, exc)
        result = "Something went wrong saving your session. Please contact your coach."
    finally:
        _wa_sessions.pop(phone, None)

    return result


def _handle_cancel(phone: str) -> str:
    if _wa_sessions.pop(phone, None) is None:
        return "No active session to cancel."
    return "Session discarded. Nothing was saved. Send *start* to begin again."


def _handle_message(phone: str, text: str) -> str:
    session = _wa_sessions.get(phone)
    if session is None:
        return (
            "No active session. Send *start* to begin your weekly coaching check-in.\n\n"
            + HELP_TEXT
        )
    try:
        return session.chat(text)
    except Exception as exc:
        logger.error("session.chat failed for wa_phone=%s: %s", phone, exc)
        return "I'm having trouble responding right now. Please try again in a moment."


def _route_message(phone: str, sender_name: str, text: str) -> str:
    """Dispatch the incoming message to the right handler."""
    keyword = text.strip().lower()

    if keyword in ("start", "hi", "hello", "hey"):
        return _handle_start(phone, sender_name)
    if keyword in ("done", "end", "finish"):
        return _handle_end(phone)
    if keyword == "cancel":
        return _handle_cancel(phone)
    if keyword == "help":
        return HELP_TEXT

    return _handle_message(phone, text)


# ── FastAPI routes ─────────────────────────────────────────────────────────────

@router.get(
    "/webhook",
    response_class=PlainTextResponse,
    summary="Meta webhook verification handshake",
    include_in_schema=False,
)
async def whatsapp_verify(request: Request) -> PlainTextResponse:
    """
    Meta calls this when you first register the webhook URL.
    It sends hub.mode=subscribe, hub.verify_token=<your token>, hub.challenge=<nonce>.
    We must respond with the challenge string if the verify_token matches.
    """
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    verify_token = coaching_config.whatsapp_verify_token
    if not verify_token:
        logger.error("WHATSAPP_VERIFY_TOKEN is not configured")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="WhatsApp verify token not configured.")

    if mode == "subscribe" and token == verify_token:
        logger.info("WhatsApp webhook verified successfully")
        return PlainTextResponse(content=challenge or "")

    logger.warning("WhatsApp webhook verification failed: mode=%s token_match=%s", mode, token == verify_token)
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Webhook verification failed.")


@router.post(
    "/webhook",
    summary="Receive incoming WhatsApp messages from Meta",
    include_in_schema=False,
)
async def whatsapp_receive(request: Request) -> dict:
    """
    Meta POSTs every incoming message here.
    We verify the signature, extract sender + text, dispatch to the session handler,
    and reply via the WhatsApp Cloud API.

    Always returns HTTP 200 — Meta will retry on non-200.
    """
    body_bytes = await request.body()

    # Verify webhook authenticity
    sig_header = request.headers.get("X-Hub-Signature-256", "")
    if not _verify_signature(body_bytes, sig_header):
        logger.warning("Invalid WhatsApp webhook signature — request rejected")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid signature.")

    try:
        data = json.loads(body_bytes)
    except json.JSONDecodeError:
        logger.error("Could not parse WhatsApp webhook payload")
        return {"status": "ok"}  # always 200 to Meta

    # Walk the nested Meta payload structure
    try:
        entry = data.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})

        # Ignore status updates (delivery receipts etc.)
        if "statuses" in value and "messages" not in value:
            return {"status": "ok"}

        messages = value.get("messages", [])
        if not messages:
            return {"status": "ok"}

        message = messages[0]
        msg_type = message.get("type", "")

        # Only handle plain text messages
        if msg_type != "text":
            logger.info("Ignoring non-text WhatsApp message type=%s", msg_type)
            return {"status": "ok"}

        phone = message.get("from", "")
        text = message.get("text", {}).get("body", "").strip()
        if not phone or not text:
            return {"status": "ok"}

        # Best-effort sender name
        contacts = value.get("contacts", [{}])
        sender_name = contacts[0].get("profile", {}).get("name", phone) if contacts else phone

        logger.info("WhatsApp message from=%s text=%r", phone, text[:80])
        reply = _route_message(phone, sender_name, text)
        _send_whatsapp_text(phone, reply)

    except Exception as exc:
        logger.exception("Unhandled error processing WhatsApp webhook: %s", exc)

    # Always return 200 so Meta does not retry
    return {"status": "ok"}
