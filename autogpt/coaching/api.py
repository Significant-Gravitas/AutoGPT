"""FastAPI application for the ABN Consulting AI Co-Navigator."""
from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from autogpt.coaching.config import coaching_config
from autogpt.coaching.dashboard import build_dashboard
from autogpt.coaching.models import (
    AuthResponse,
    CoachDashboard,
    GoogleAuthRequest,
    KeyResultRequest,
    LoginRequest,
    Objective,
    ObjectiveRequest,
    OKRStatus,
    PastSession,
    RegisterRequest,
    SessionSummary,
    StatusUpdateRequest,
    UserProfile,
)
from autogpt.coaching.session import CoachingSession
from autogpt.coaching.storage import (
    get_past_sessions,
    get_user_objectives,
    get_user_profile,
    google_auth,
    load_session,
    login_user,
    register_user,
    save_session,
    set_kr_status,
    set_objective_status,
    upsert_master_kr,
    upsert_objective,
)

app = FastAPI(
    title="ABN Consulting AI Co-Navigator",
    description="Executive change management coaching API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://*.wix.com", "https://*.wixsite.com", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API-key guard (Wix → API server auth) ────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def verify_api_key(key: str = Security(api_key_header)) -> str:
    if not coaching_config.api_key:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Server API key not configured.")
    if key != coaching_config.api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Invalid API key.")
    return key


# ── In-memory active session store ───────────────────────────────────────────

_active_sessions: Dict[str, CoachingSession] = {}


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.post("/auth/register", response_model=AuthResponse, summary="Register a new user")
def auth_register(req: RegisterRequest, _: str = Depends(verify_api_key)) -> AuthResponse:
    try:
        user = register_user(name=req.name, email=req.email, password=req.password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return AuthResponse(user_id=user.user_id, name=user.name, email=user.email)


@app.post("/auth/login", response_model=AuthResponse, summary="Login with email and password")
def auth_login(req: LoginRequest, _: str = Depends(verify_api_key)) -> AuthResponse:
    try:
        user = login_user(email=req.email, password=req.password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))
    return AuthResponse(user_id=user.user_id, name=user.name, email=user.email)


@app.post("/auth/google", response_model=AuthResponse,
          summary="Register or login via Google OAuth (call after Wix handles OAuth)")
def auth_google(req: GoogleAuthRequest, _: str = Depends(verify_api_key)) -> AuthResponse:
    user = google_auth(google_id=req.google_id, name=req.name, email=req.email)
    return AuthResponse(user_id=user.user_id, name=user.name, email=user.email)


# ── User profile ──────────────────────────────────────────────────────────────

@app.get("/users/{user_id}/profile", response_model=UserProfile, summary="Get user profile")
def get_profile(user_id: str, _: str = Depends(verify_api_key)) -> UserProfile:
    user = get_user_profile(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return user


# ── Objectives ────────────────────────────────────────────────────────────────

@app.get("/users/{user_id}/objectives", response_model=List[Objective],
         summary="Get user's active objectives with key results")
def list_objectives(user_id: str, _: str = Depends(verify_api_key)) -> List[Objective]:
    return get_user_objectives(user_id)


@app.post("/users/{user_id}/objectives", response_model=Objective,
          summary="Create or update an objective")
def create_or_update_objective(
    user_id: str,
    req: ObjectiveRequest,
    _: str = Depends(verify_api_key),
) -> Objective:
    return upsert_objective(
        user_id=user_id,
        title=req.title,
        description=req.description,
        objective_id=req.objective_id,
    )


@app.put("/objectives/{objective_id}/status", response_model=dict,
         summary="Set objective status (active / archived / on_hold)")
def update_objective_status(
    objective_id: str,
    req: StatusUpdateRequest,
    _: str = Depends(verify_api_key),
) -> dict:
    set_objective_status(objective_id, req.status)
    return {"objective_id": objective_id, "status": req.status.value}


# ── Key Results ───────────────────────────────────────────────────────────────

@app.post("/users/{user_id}/key-results", response_model=dict,
          summary="Create or update a key result")
def create_or_update_kr(
    user_id: str,
    req: KeyResultRequest,
    _: str = Depends(verify_api_key),
) -> dict:
    kr = upsert_master_kr(
        objective_id=req.objective_id,
        user_id=user_id,
        description=req.description,
        current_pct=req.current_pct,
        kr_id=req.kr_id,
    )
    return {"kr_id": kr.kr_id, "objective_id": kr.objective_id, "description": kr.description,
            "current_pct": kr.current_pct}


@app.put("/key-results/{kr_id}/status", response_model=dict,
         summary="Set key result status (active / archived / on_hold)")
def update_kr_status(
    kr_id: str,
    req: StatusUpdateRequest,
    _: str = Depends(verify_api_key),
) -> dict:
    set_kr_status(kr_id, req.status)
    return {"kr_id": kr_id, "status": req.status.value}


# ── History ───────────────────────────────────────────────────────────────────

@app.get("/users/{user_id}/history", response_model=List[PastSession],
         summary="Get past session highlights for a user")
def user_history(user_id: str, _: str = Depends(verify_api_key)) -> List[PastSession]:
    return get_past_sessions(user_id=user_id, limit=10)


# ── Coaching sessions ─────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    client_id: str
    client_name: str
    user_id: Optional[str] = None  # links session to a registered user


class StartSessionResponse(BaseModel):
    session_id: str
    message: str


class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    session_id: str
    reply: str


@app.post("/coaching/session/start", response_model=StartSessionResponse,
          summary="Start a new coaching session")
def start_session(
    req: StartSessionRequest,
    _: str = Depends(verify_api_key),
) -> StartSessionResponse:
    # Load user context when user_id is provided
    objectives = []
    past_sessions = []
    user_name = req.client_name

    if req.user_id:
        profile = get_user_profile(req.user_id)
        if profile:
            user_name = profile.name
        objectives = get_user_objectives(req.user_id)
        past_sessions = get_past_sessions(req.user_id, limit=3)

    session = CoachingSession(
        client_id=req.client_id,
        client_name=user_name,
        user_id=req.user_id,
        objectives=objectives,
        past_sessions=past_sessions,
    )
    _active_sessions[session.session_id] = session

    opening = session.open()
    return StartSessionResponse(session_id=session.session_id, message=opening)


@app.post("/coaching/session/{session_id}/message", response_model=MessageResponse,
          summary="Send a message in an active session")
def send_message(
    session_id: str,
    req: MessageRequest,
    _: str = Depends(verify_api_key),
) -> MessageResponse:
    session = _active_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Active session '{session_id}' not found.")
    reply = session.chat(req.message)
    return MessageResponse(session_id=session_id, reply=reply)


@app.post("/coaching/session/{session_id}/end", response_model=SessionSummary,
          summary="End session — extract summary + OKR changes, save to Supabase")
def end_session(
    session_id: str,
    _: str = Depends(verify_api_key),
) -> SessionSummary:
    session = _active_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Active session '{session_id}' not found.")
    summary = session.extract_summary()
    save_session(summary)
    del _active_sessions[session_id]
    return summary


@app.get("/coaching/session/{session_id}", response_model=SessionSummary,
         summary="Load a saved session from Supabase")
def get_session(session_id: str, _: str = Depends(verify_api_key)) -> SessionSummary:
    summary = load_session(session_id)
    if summary is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Session '{session_id}' not found.")
    return summary


@app.get("/coaching/dashboard", response_model=CoachDashboard,
         summary="Coach dashboard — latest status for all clients")
def get_dashboard(_: str = Depends(verify_api_key)) -> CoachDashboard:
    return build_dashboard()


@app.get("/health", summary="Health check")
def health() -> dict:
    return {"status": "ok", "service": "ABN Co-Navigator API"}
