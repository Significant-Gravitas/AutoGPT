"""Pydantic data models for the ABN Consulting AI Co-Navigator."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, field_validator


class AlertLevel(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class NavigationStatus(str, Enum):
    CLEAR = "clear"
    CHOPPY = "choppy"
    STORMY = "stormy"


class KeyResult(BaseModel):
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
    reported_at: datetime = None
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
    timestamp: datetime
    weekly_log: WeeklyLog
    alerts: Alert
    summary_for_coach: str


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
