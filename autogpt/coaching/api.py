"""FastAPI application for the ABN Consulting AI Co-Navigator."""
from __future__ import annotations

from typing import Dict

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from autogpt.coaching.config import coaching_config
from autogpt.coaching.dashboard import build_dashboard
from autogpt.coaching.models import CoachDashboard, SessionSummary
from autogpt.coaching.session import CoachingSession
from autogpt.coaching.storage import load_session, save_session

app = FastAPI(
    title="ABN Consulting AI Co-Navigator",
    description="Executive change management coaching API",
    version="1.0.0",
)

# CORS — allow Wix origin and localhost for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://*.wix.com", "https://*.wixsite.com", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Authentication ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def verify_api_key(key: str = Security(api_key_header)) -> str:
    if not coaching_config.api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server API key not configured.",
        )
    if key != coaching_config.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
    return key


# --- In-memory active session store ---
_active_sessions: Dict[str, CoachingSession] = {}


# --- Request / Response models ---
class StartSessionRequest(BaseModel):
    client_id: str
    client_name: str


class StartSessionResponse(BaseModel):
    session_id: str
    message: str


class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    session_id: str
    reply: str


# --- Endpoints ---

@app.post(
    "/coaching/session/start",
    response_model=StartSessionResponse,
    summary="Start a new coaching session",
)
def start_session(
    req: StartSessionRequest,
    _: str = Depends(verify_api_key),
) -> StartSessionResponse:
    session = CoachingSession(client_id=req.client_id, client_name=req.client_name)
    _active_sessions[session.session_id] = session

    # Send the opening message from Navigator
    opening = session.chat(
        f"Hello! I'm {req.client_name}. I'm ready for our weekly check-in."
    )
    return StartSessionResponse(session_id=session.session_id, message=opening)


@app.post(
    "/coaching/session/{session_id}/message",
    response_model=MessageResponse,
    summary="Send a message in an active session",
)
def send_message(
    session_id: str,
    req: MessageRequest,
    _: str = Depends(verify_api_key),
) -> MessageResponse:
    session = _active_sessions.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Active session '{session_id}' not found. It may have already ended.",
        )
    reply = session.chat(req.message)
    return MessageResponse(session_id=session_id, reply=reply)


@app.post(
    "/coaching/session/{session_id}/end",
    response_model=SessionSummary,
    summary="End a session — extract summary and save to Supabase",
)
def end_session(
    session_id: str,
    _: str = Depends(verify_api_key),
) -> SessionSummary:
    session = _active_sessions.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Active session '{session_id}' not found.",
        )

    summary = session.extract_summary()
    save_session(summary)
    del _active_sessions[session_id]
    return summary


@app.get(
    "/coaching/session/{session_id}",
    response_model=SessionSummary,
    summary="Load a saved session from Supabase",
)
def get_session(
    session_id: str,
    _: str = Depends(verify_api_key),
) -> SessionSummary:
    summary = load_session(session_id)
    if summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found in database.",
        )
    return summary


@app.get(
    "/coaching/dashboard",
    response_model=CoachDashboard,
    summary="Get the coach dashboard — latest status for all clients",
)
def get_dashboard(_: str = Depends(verify_api_key)) -> CoachDashboard:
    return build_dashboard()


@app.get("/health", summary="Health check")
def health() -> dict:
    return {"status": "ok", "service": "ABN Co-Navigator API"}
