"""
Journal data models for the Change Navigator weekly check-in.

Each JournalEntry maps directly to the fields of the physical/PDF journal
used in the coaching program. Structured output from the AI is validated
against these models before being persisted or sent to the coach.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from typing import List, Optional


class JournalStatus(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    SENT = "sent"


@dataclass
class KeyResult:
    """A single Key Result tracked within an OKR cycle."""

    name: str
    target: str
    progress_pct: int = 0  # 0-100
    notes: str = ""

    def __post_init__(self) -> None:
        if not (0 <= self.progress_pct <= 100):
            raise ValueError(
                f"progress_pct must be between 0 and 100, got {self.progress_pct}"
            )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class JournalEntry:
    """
    Weekly journal entry for a Change Navigator coaching session.

    Fields mirror the structured sections of the paper/PDF journal:
      - Identity & period
      - Central goal (set with the coach)
      - Key Results (OKR tracking)
      - Obstacles & environmental risks
      - Reflection on the week's leadership inspiration
      - Coach-facing summary notes
    """

    coachee_name: str
    week_number: int
    entry_date: date = field(default_factory=date.today)
    central_goal: str = ""
    key_results: List[KeyResult] = field(default_factory=list)
    obstacles: List[str] = field(default_factory=list)
    inspiration_reflection: str = ""
    coach_notes: str = ""
    status: JournalStatus = JournalStatus.DRAFT

    # ---------------------------------------------------------------------------
    # Serialisation helpers
    # ---------------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entry_date"] = self.entry_date.isoformat()
        d["status"] = self.status.value
        d["key_results"] = [kr.to_dict() for kr in self.key_results]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "JournalEntry":
        data = dict(data)
        if "entry_date" in data and isinstance(data["entry_date"], str):
            data["entry_date"] = date.fromisoformat(data["entry_date"])
        if "status" in data:
            data["status"] = JournalStatus(data["status"])
        if "key_results" in data:
            data["key_results"] = [
                KeyResult(**kr) for kr in data["key_results"]
            ]
        return cls(**data)

    # ---------------------------------------------------------------------------
    # Human-readable summary (shown to coachee for approval before sending)
    # ---------------------------------------------------------------------------

    def summary_text(self) -> str:
        lines = [
            f"=== Change Navigator Weekly Journal — Week {self.week_number} ===",
            f"Coachee  : {self.coachee_name}",
            f"Date     : {self.entry_date.isoformat()}",
            f"Status   : {self.status.value.upper()}",
            "",
            f"Central Goal:\n  {self.central_goal or '—'}",
            "",
            "Key Results:",
        ]
        for i, kr in enumerate(self.key_results, 1):
            lines.append(
                f"  KR{i}: {kr.name} → {kr.progress_pct}%"
                + (f" | {kr.notes}" if kr.notes else "")
            )
        if not self.key_results:
            lines.append("  (none recorded)")

        lines += [
            "",
            "Obstacles & Risks:",
        ]
        for obs in self.obstacles:
            lines.append(f"  • {obs}")
        if not self.obstacles:
            lines.append("  (none recorded)")

        lines += [
            "",
            f"Inspiration Reflection:\n  {self.inspiration_reflection or '—'}",
            "",
            f"Coach Notes:\n  {self.coach_notes or '—'}",
        ]
        return "\n".join(lines)
