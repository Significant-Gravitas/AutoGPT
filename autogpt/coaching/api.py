"""FastAPI application for the ABN Consulting AI Co-Navigator."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from urllib.parse import urlencode

import requests as http_requests
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

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

async def _session_cleanup_loop() -> None:
    """Background task: remove expired in-memory sessions every 10 minutes."""
    while True:
        await asyncio.sleep(600)
        expired = [sid for sid, s in list(_active_sessions.items()) if s.is_expired()]
        for sid in expired:
            logger.info("Evicting expired session %s", sid)
            _active_sessions.pop(sid, None)


@asynccontextmanager
async def lifespan(application: FastAPI):
    task = asyncio.create_task(_session_cleanup_loop())
    yield
    task.cancel()


app = FastAPI(
    title="ABN Consulting AI Co-Navigator",
    description="Executive change management coaching API",
    version="2.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Mount the WhatsApp Business webhook router
from autogpt.coaching.whatsapp_bot import router as whatsapp_router  # noqa: E402
app.include_router(whatsapp_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.ben-nesher.com", "https://*.wix.com", "https://*.wixsite.com", "http://localhost:3000"],
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
@limiter.limit("10/minute")
def auth_register(request: Request, req: RegisterRequest, _: str = Depends(verify_api_key)) -> AuthResponse:
    try:
        user = register_user(name=req.name, email=req.email, password=req.password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return AuthResponse(user_id=user.user_id, name=user.name, email=user.email)


@app.post("/auth/login", response_model=AuthResponse, summary="Login with email and password")
@limiter.limit("20/minute")
def auth_login(request: Request, req: LoginRequest, _: str = Depends(verify_api_key)) -> AuthResponse:
    try:
        user = login_user(email=req.email, password=req.password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))
    return AuthResponse(user_id=user.user_id, name=user.name, email=user.email)


@app.post("/auth/google", response_model=AuthResponse,
          summary="Register or login via Google OAuth (call after Wix handles OAuth)")
@limiter.limit("20/minute")
def auth_google(request: Request, req: GoogleAuthRequest, _: str = Depends(verify_api_key)) -> AuthResponse:
    user = google_auth(google_id=req.google_id, name=req.name, email=req.email)
    return AuthResponse(user_id=user.user_id, name=user.name, email=user.email)


@app.get(
    "/auth/google/url",
    summary="Redirect the user's browser to Google's OAuth consent screen",
    response_class=RedirectResponse,
)
def google_oauth_start(
    redirect_to: str = Query(
        ...,
        description="The Wix page URL to return the user to after authentication "
                    "(e.g. https://yoursite.com/dashboard)",
    ),
) -> RedirectResponse:
    """
    Wix links the 'Sign in with Google' button directly to this endpoint.
    The user's browser is redirected to Google's consent screen.
    After consent, Google calls /auth/google/callback which then sends the
    user back to the *redirect_to* URL with user_id, name, and email as query params.
    """
    if not coaching_config.google_client_id or not coaching_config.google_redirect_uri:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured on this server.",
        )

    # Encode the Wix return URL inside the `state` param so we can retrieve it in the callback
    state = base64.urlsafe_b64encode(redirect_to.encode()).decode()

    params = {
        "client_id": coaching_config.google_client_id,
        "redirect_uri": coaching_config.google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "online",
        "state": state,
        "prompt": "select_account",
    }
    google_auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
    return RedirectResponse(url=google_auth_url, status_code=302)


@app.get(
    "/auth/google/callback",
    summary="Google OAuth callback — exchanges code, creates/finds user, redirects to Wix",
    response_class=RedirectResponse,
)
def google_oauth_callback(
    code: str = Query(..., description="Authorization code from Google"),
    state: str = Query(..., description="Base64-encoded Wix return URL"),
    error: Optional[str] = Query(None, description="Error from Google (e.g. access_denied)"),
) -> RedirectResponse:
    """
    Google redirects here after the user consents.
    This endpoint is the value you must enter as 'Authorized redirect URI'
    in the Google Cloud Console OAuth 2.0 client settings.
    """
    # Decode the Wix return URL
    try:
        redirect_to = base64.urlsafe_b64decode(state.encode()).decode()
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid state parameter.")

    if error:
        return RedirectResponse(url=f"{redirect_to}?error={error}", status_code=302)

    # Exchange authorization code for tokens
    token_resp = http_requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": coaching_config.google_client_id,
            "client_secret": coaching_config.google_client_secret,
            "redirect_uri": coaching_config.google_redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=10,
    )
    if token_resp.status_code != 200:
        return RedirectResponse(url=f"{redirect_to}?error=token_exchange_failed", status_code=302)

    access_token = token_resp.json().get("access_token")

    # Fetch user info from Google
    userinfo_resp = http_requests.get(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    if userinfo_resp.status_code != 200:
        return RedirectResponse(url=f"{redirect_to}?error=userinfo_failed", status_code=302)

    userinfo = userinfo_resp.json()
    google_id = userinfo.get("sub")
    name = userinfo.get("name", "")
    email = userinfo.get("email", "")

    if not google_id or not email:
        return RedirectResponse(url=f"{redirect_to}?error=incomplete_profile", status_code=302)

    user = google_auth(google_id=google_id, name=name, email=email)

    # Return user to Wix with identity attached as query params
    params = urlencode({"user_id": user.user_id, "name": user.name, "email": user.email})
    return RedirectResponse(url=f"{redirect_to}?{params}", status_code=302)


@app.get("/auth/google/config", summary="Show the redirect URI to register in Google Cloud Console")
def google_oauth_config(_: str = Depends(verify_api_key)) -> dict:
    return {
        "google_redirect_uri": coaching_config.google_redirect_uri or "(not configured — set GOOGLE_REDIRECT_URI env var)",
        "instructions": (
            "Copy the value of 'google_redirect_uri' and paste it into "
            "Google Cloud Console → APIs & Services → Credentials → "
            "your OAuth 2.0 Client ID → Authorized redirect URIs."
        ),
    }


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
@limiter.limit("10/minute")
def start_session(
    request: Request,
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
@limiter.limit("60/minute")
def send_message(
    request: Request,
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
