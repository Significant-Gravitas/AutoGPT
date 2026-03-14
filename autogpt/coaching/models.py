"""Pydantic data models for the ABN Consulting AI Co-Navigator."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class AlertLevel(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class NavigationStatus(str, Enum):
    CLEAR = "clear"
    CHOPPY = "choppy"
    STORMY = "stormy"


class OKRStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    ON_HOLD = "on_hold"


# ── User / Auth ───────────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    user_id: str
    name: str
    email: str


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class GoogleAuthRequest(BaseModel):
    """Called after Wix completes Google OAuth; provides the resolved identity."""
    google_id: str
    name: str
    email: str


class AuthResponse(BaseModel):
    user_id: str
    name: str
    email: str


# ── OKR Master Plan ───────────────────────────────────────────────────────────

class MasterKeyResult(BaseModel):
    """A Key Result in the user's ongoing OKR plan (not session-specific)."""
    kr_id: str
    objective_id: str
    description: str
    status: OKRStatus = OKRStatus.ACTIVE
    current_pct: int = Field(default=0, ge=0, le=100)


class Objective(BaseModel):
    """A user's objective with its associated key results."""
    objective_id: str
    user_id: str
    title: str
    description: str = ""
    status: OKRStatus = OKRStatus.ACTIVE
    key_results: List[MasterKeyResult] = []


class ObjectiveRequest(BaseModel):
    title: str
    description: str = ""
    objective_id: Optional[str] = None  # present → update; absent → create


class KeyResultRequest(BaseModel):
    objective_id: str
    description: str
    current_pct: int = Field(default=0, ge=0, le=100)
    kr_id: Optional[str] = None  # present → update; absent → create


class StatusUpdateRequest(BaseModel):
    status: OKRStatus


# ── Session / Weekly Log ──────────────────────────────────────────────────────

class KeyResult(BaseModel):
    """Session-specific KR snapshot."""
    kr_id: int
    description: str
    status_pct: int  # 0–100
    status_color: str = ""

    @field_validator("status_pct")
    @classmethod
    def clamp_pct(cls, v: int) -> int:
        return max(0, min(100, v))

    def model_post_init(self, __context) -> None:
        if not self.status_color:
            if self.status_pct >= 70:
                self.status_color = "green"
            elif self.status_pct >= 40:
                self.status_color = "yellow"
            else:
                self.status_color = "red"


class Obstacle(BaseModel):
    description: str
    reported_at: Optional[datetime] = None
    resolved: bool = False

    def model_post_init(self, __context) -> None:
        if self.reported_at is None:
            self.reported_at = datetime.utcnow()


class WeeklyLog(BaseModel):
    focus_goal: str = ""
    key_results: List[KeyResult] = []
    environmental_changes: str = ""
    obstacles: List[Obstacle] = []
    mood_indicator: str = ""

    def avg_kr_pct(self) -> float:
        if not self.key_results:
            return 0.0
        return sum(kr.status_pct for kr in self.key_results) / len(self.key_results)

    def has_unresolved_obstacles(self) -> bool:
        return any(not o.resolved for o in self.obstacles)


class Alert(BaseModel):
    level: AlertLevel
    reason: str


class SessionSummary(BaseModel):
    session_id: str
    client_id: str
    client_name: str
    user_id: Optional[str] = None
    timestamp: datetime
    weekly_log: WeeklyLog
    alerts: Alert
    summary_for_coach: str
    okr_changes: List[Dict[str, Any]] = []  # structured OKR mutations to apply


# ── History ───────────────────────────────────────────────────────────────────

class PastSession(BaseModel):
    session_id: str
    timestamp: str
    alert_level: str
    summary_for_coach: str


# ── Dashboard ─────────────────────────────────────────────────────────────────

class ClientStatus(BaseModel):
    client_id: str
    name: str
    navigation_status: NavigationStatus
    key_results_avg: float
    active_alerts: List[Alert] = []
    last_session: Optional[datetime] = None


class CoachDashboard(BaseModel):
    generated_at: datetime
    clients: List[ClientStatus] = []
