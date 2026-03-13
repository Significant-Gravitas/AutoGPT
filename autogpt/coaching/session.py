"""CoachingSession: manages a single Co-Navigator conversation with a client."""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from autogpt.coaching.config import coaching_config
from autogpt.coaching.models import (
    Alert,
    AlertLevel,
    KeyResult,
    Obstacle,
    Objective,
    PastSession,
    SessionSummary,
    WeeklyLog,
)
from autogpt.coaching.prompts import SUMMARY_EXTRACTION_PROMPT, build_navigator_system_prompt

Message = dict  # {"role": str, "content": str}


class CoachingSession:
    """Manages one coaching conversation session with a client."""

    def __init__(
        self,
        client_id: str,
        client_name: str,
        user_id: Optional[str] = None,
        objectives: Optional[List[Objective]] = None,
        past_sessions: Optional[List[PastSession]] = None,
    ) -> None:
        self.session_id: str = str(uuid.uuid4())
        self.client_id: str = client_id
        self.client_name: str = client_name
        self.user_id: Optional[str] = user_id
        self.objectives: List[Objective] = objectives or []
        self.timestamp: datetime = datetime.utcnow()
        self.full_message_history: List[Message] = []
        self._system_prompt: str = build_navigator_system_prompt(
            coach_name=coaching_config.coach_name,
            calendly_url=coaching_config.coach_calendly_url,
            objectives=objectives,
            past_sessions=past_sessions,
        )

    def open(self) -> str:
        """Generate the opening Navigator message for this session."""
        if self.user_id and self.objectives:
            opener = (
                f"Hello {self.client_name}! Welcome back to your weekly check-in. "
                f"Let's start by reviewing your current objectives."
            )
        elif self.user_id and not self.objectives:
            opener = (
                f"Hello {self.client_name}! Welcome to the coaching program. "
                f"Since this is your first session, let's begin by setting up your objectives and key results."
            )
        else:
            opener = f"Hello {self.client_name}! I'm ready for our weekly check-in."

        return self.chat(opener)

    def chat(self, user_message: str) -> str:
        """Send a user message and get the Navigator's reply."""
        self.full_message_history.append({"role": "user", "content": user_message})

        messages: List[Message] = [
            {"role": "system", "content": self._system_prompt},
        ] + self.full_message_history

        from autogpt.coaching.llm import chat_completion
        reply = chat_completion(
            messages=messages,
            model=coaching_config.llm_model,
            temperature=coaching_config.llm_temperature,
        )
        self.full_message_history.append({"role": "assistant", "content": reply})
        return reply

    def extract_summary(self) -> SessionSummary:
        """Ask the LLM to produce a structured JSON summary + OKR changes, then parse."""
        from autogpt.coaching.llm import chat_completion

        extraction_messages: List[Message] = [
            {"role": "system", "content": self._system_prompt},
        ] + self.full_message_history + [
            {"role": "user", "content": SUMMARY_EXTRACTION_PROMPT}
        ]

        raw = chat_completion(
            messages=extraction_messages,
            model=coaching_config.llm_model,
            temperature=0.0,
        )

        weekly_log, summary_text = self._parse_summary_json(raw)
        okr_changes = self._parse_okr_changes(raw)
        alert = self._compute_alerts(weekly_log)

        return SessionSummary(
            session_id=self.session_id,
            client_id=self.client_id,
            client_name=self.client_name,
            user_id=self.user_id,
            timestamp=self.timestamp,
            weekly_log=weekly_log,
            alerts=alert,
            summary_for_coach=summary_text,
            okr_changes=okr_changes,
        )

    # ── Parsers ────────────────────────────────────────────────────────────────

    def _parse_summary_json(self, raw: str) -> tuple[WeeklyLog, str]:
        match = re.search(
            r"\[SESSION_SUMMARY_JSON\](.*?)\[/SESSION_SUMMARY_JSON\]",
            raw,
            re.DOTALL,
        )
        json_text = match.group(1).strip() if match else raw.strip()

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            return WeeklyLog(), "Could not extract structured summary from this session."

        key_results = [
            KeyResult(
                kr_id=kr.get("kr_id", i + 1),
                description=kr.get("description", ""),
                status_pct=int(kr.get("status_pct", 0)),
            )
            for i, kr in enumerate(data.get("key_results", []))
        ]

        obstacles = [
            Obstacle(
                description=obs.get("description", ""),
                resolved=obs.get("resolved", False),
            )
            for obs in data.get("obstacles", [])
        ]

        weekly_log = WeeklyLog(
            focus_goal=data.get("focus_goal", ""),
            key_results=key_results,
            environmental_changes=data.get("environmental_changes", ""),
            obstacles=obstacles,
            mood_indicator=data.get("mood_indicator", ""),
        )

        summary_text = data.get("summary_for_coach", "")
        return weekly_log, summary_text

    def _parse_okr_changes(self, raw: str) -> List[Dict[str, Any]]:
        """Extract the OKR_CHANGES_JSON block from the LLM response."""
        match = re.search(
            r"\[OKR_CHANGES_JSON\](.*?)\[/OKR_CHANGES_JSON\]",
            raw,
            re.DOTALL,
        )
        if not match:
            return []
        try:
            data = json.loads(match.group(1).strip())
            return data.get("okr_changes", [])
        except (json.JSONDecodeError, AttributeError):
            return []

    def _compute_alerts(self, weekly_log: WeeklyLog) -> Alert:
        avg = weekly_log.avg_kr_pct()
        has_obstacles = weekly_log.has_unresolved_obstacles()

        if avg < coaching_config.alert_red_threshold:
            return Alert(
                level=AlertLevel.RED,
                reason=f"Average KR completion critically low ({avg:.0f}%). Immediate attention needed.",
            )
        if has_obstacles or avg < coaching_config.alert_yellow_threshold:
            reason_parts = []
            if has_obstacles:
                unresolved = [o.description for o in weekly_log.obstacles if not o.resolved]
                reason_parts.append(f"Unresolved obstacles: {'; '.join(unresolved[:2])}")
            if avg < coaching_config.alert_yellow_threshold:
                reason_parts.append(f"KR average at {avg:.0f}%")
            return Alert(level=AlertLevel.YELLOW, reason=". ".join(reason_parts))

        return Alert(level=AlertLevel.GREEN, reason=f"Good progress — KR average at {avg:.0f}%.")

    def conversation_as_json(self) -> list:
        return self.full_message_history
