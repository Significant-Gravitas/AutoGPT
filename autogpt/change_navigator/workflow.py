"""
CheckInWorkflow — drives the five-stage weekly check-in conversation.

Stage flow:
  opening → key_results (multi-turn) → obstacles → reflection → approval

The workflow is language-aware: if the user's first message is in Hebrew,
all agent questions are sent in Hebrew, and vice versa.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from autogpt.change_navigator.journal import JournalEntry, JournalStatus, KeyResult
from autogpt.change_navigator.prompts import (
    APPROVAL_MESSAGE_EN,
    APPROVAL_MESSAGE_HE,
    CHECKIN_QUESTIONS,
    EXTRACTION_PROMPT,
)

logger = logging.getLogger(__name__)


class Stage(Enum):
    OPENING = auto()
    KEY_RESULTS = auto()
    OBSTACLES = auto()
    REFLECTION = auto()
    APPROVAL = auto()
    DONE = auto()


STAGE_ORDER = [
    Stage.OPENING,
    Stage.KEY_RESULTS,
    Stage.OBSTACLES,
    Stage.REFLECTION,
    Stage.APPROVAL,
    Stage.DONE,
]


def _detect_hebrew(text: str) -> bool:
    """Return True if the text contains Hebrew characters."""
    return any("\u05d0" <= ch <= "\u05ea" for ch in text)


@dataclass
class CheckInWorkflow:
    """
    Stateful workflow for a single weekly check-in session.

    Usage
    -----
    workflow = CheckInWorkflow(coachee_name="Jane Doe", week_number=12)
    question = workflow.start()          # Returns opening question
    while not workflow.is_done():
        answer = <get user input>
        reply  = workflow.advance(answer)
        print(reply)
    entry = workflow.journal_entry        # Finalised JournalEntry
    """

    coachee_name: str
    week_number: int
    entry_date: date = field(default_factory=date.today)

    # Internal state
    _stage: Stage = field(default=Stage.OPENING, init=False)
    _transcript: List[Dict[str, str]] = field(default_factory=list, init=False)
    _kr_buffer: List[dict] = field(default_factory=list, init=False)
    _kr_index: int = field(default=0, init=False)
    _hebrew: bool = field(default=False, init=False)
    _journal: Optional[JournalEntry] = field(default=None, init=False)

    # Injected LLM callable: (system_prompt, messages) -> str
    # Replace with autogpt.llm_utils.create_chat_completion or similar.
    _llm_call: Optional[object] = field(default=None, init=False)

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def set_llm(self, llm_callable) -> None:
        """Inject the LLM callable used for transcript extraction."""
        self._llm_call = llm_callable

    def start(self) -> str:
        """Return the first question (language auto-detected on first reply)."""
        q = CHECKIN_QUESTIONS[0]
        return q["question_en"]  # Language detected after first reply

    def advance(self, user_input: str) -> str:
        """
        Process the user's answer and return the next agent message.

        Parameters
        ----------
        user_input : str
            The executive's latest reply.

        Returns
        -------
        str
            The agent's next question, confirmation, or summary.
        """
        # Detect language on first interaction
        if not self._transcript:
            self._hebrew = _detect_hebrew(user_input)

        self._transcript.append({"role": "user", "content": user_input})

        if self._stage == Stage.OPENING:
            return self._handle_opening(user_input)
        if self._stage == Stage.KEY_RESULTS:
            return self._handle_key_results(user_input)
        if self._stage == Stage.OBSTACLES:
            return self._handle_obstacles(user_input)
        if self._stage == Stage.REFLECTION:
            return self._handle_reflection(user_input)
        if self._stage == Stage.APPROVAL:
            return self._handle_approval(user_input)

        return ""  # DONE

    def is_done(self) -> bool:
        return self._stage == Stage.DONE

    @property
    def journal_entry(self) -> Optional[JournalEntry]:
        return self._journal

    # ---------------------------------------------------------------------------
    # Stage handlers
    # ---------------------------------------------------------------------------

    def _handle_opening(self, answer: str) -> str:
        self._central_goal = answer
        self._stage = Stage.KEY_RESULTS
        q = CHECKIN_QUESTIONS[1]
        reply = q["question_he"] if self._hebrew else q["question_en"]
        self._transcript.append({"role": "assistant", "content": reply})
        return reply

    def _handle_key_results(self, answer: str) -> str:
        """
        Multi-turn KR collection.  The agent keeps asking for the next KR
        until the user says 'done' / 'סיימתי'.
        """
        done_words = {"done", "finish", "finished", "סיימתי", "סיום", "זהו"}
        if answer.strip().lower() in done_words or self._kr_index >= 5:
            # Move to obstacles
            self._stage = Stage.OBSTACLES
            q = CHECKIN_QUESTIONS[2]
            reply = q["question_he"] if self._hebrew else q["question_en"]
            self._transcript.append({"role": "assistant", "content": reply})
            return reply

        # Parse KR from free-text (best-effort)
        kr = self._parse_kr(answer, index=self._kr_index)
        self._kr_buffer.append(kr)
        self._kr_index += 1

        if self._hebrew:
            follow_up = (
                f"רשמתי: {kr['name']} — {kr['progress_pct']}%. "
                "אם יש KR נוסף, תאר אותו. אחרת, כתוב 'סיימתי'."
            )
        else:
            follow_up = (
                f"Got it: {kr['name']} at {kr['progress_pct']}%. "
                "If there's another KR, describe it. Otherwise reply 'done'."
            )
        self._transcript.append({"role": "assistant", "content": follow_up})
        return follow_up

    def _handle_obstacles(self, answer: str) -> str:
        self._obstacles_raw = answer
        self._stage = Stage.REFLECTION
        q = CHECKIN_QUESTIONS[3]
        reply = q["question_he"] if self._hebrew else q["question_en"]
        self._transcript.append({"role": "assistant", "content": reply})
        return reply

    def _handle_reflection(self, answer: str) -> str:
        self._reflection_raw = answer
        self._stage = Stage.APPROVAL

        # Build the journal entry using LLM extraction (or heuristic fallback)
        self._journal = self._build_journal()

        summary = self._journal.summary_text()
        template = APPROVAL_MESSAGE_HE if self._hebrew else APPROVAL_MESSAGE_EN
        reply = template.format(summary=summary)
        self._transcript.append({"role": "assistant", "content": reply})
        return reply

    def _handle_approval(self, answer: str) -> str:
        approve_words = {"send", "שלח", "approve", "yes", "כן", "אישור", "ok"}
        if answer.strip().lower() in approve_words:
            self._journal.status = JournalStatus.APPROVED
            self._stage = Stage.DONE
            reply = (
                "היומן אושר ונשלח לעדי. להתראות בפגישה הבאה! ✓"
                if self._hebrew
                else "Journal approved and sent to Adi. See you at the next session! ✓"
            )
        else:
            # The user wants to change something — naive approach: re-extract
            self._transcript.append({"role": "user", "content": answer})
            if self._llm_call:
                self._journal = self._build_journal()
            reply = (
                "עדכנתי את היומן. הנה הגרסה המתוקנת:\n\n"
                + self._journal.summary_text()
                + "\n\nהאם לשלוח? (כתוב 'שלח' לאישור)"
                if self._hebrew
                else "I've updated the journal. Here is the revised version:\n\n"
                + self._journal.summary_text()
                + "\n\nShall I send it? (Reply 'send' to approve)"
            )
        self._transcript.append({"role": "assistant", "content": reply})
        return reply

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _build_journal(self) -> JournalEntry:
        """
        Build a JournalEntry from the transcript.

        Uses LLM extraction when available; falls back to the buffered
        heuristic values collected during the workflow.
        """
        if self._llm_call:
            return self._llm_extract_journal()
        return self._heuristic_journal()

    def _heuristic_journal(self) -> JournalEntry:
        """Build journal from directly collected workflow fields."""
        key_results = [
            KeyResult(
                name=kr.get("name", f"KR{i+1}"),
                target=kr.get("target", ""),
                progress_pct=kr.get("progress_pct", 0),
                notes=kr.get("notes", ""),
            )
            for i, kr in enumerate(self._kr_buffer)
        ]
        obstacles = [
            o.strip()
            for o in getattr(self, "_obstacles_raw", "").split(",")
            if o.strip()
        ]
        return JournalEntry(
            coachee_name=self.coachee_name,
            week_number=self.week_number,
            entry_date=self.entry_date,
            central_goal=getattr(self, "_central_goal", ""),
            key_results=key_results,
            obstacles=obstacles,
            inspiration_reflection=getattr(self, "_reflection_raw", ""),
            coach_notes="",
        )

    def _llm_extract_journal(self) -> JournalEntry:
        """
        Ask the LLM to parse the full transcript into structured journal JSON.
        Falls back to heuristic on any error.
        """
        transcript_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self._transcript
        )
        prompt = EXTRACTION_PROMPT.format(transcript=transcript_text)
        try:
            raw = self._llm_call(prompt)
            data = json.loads(raw)
            key_results = [KeyResult(**kr) for kr in data.get("key_results", [])]
            return JournalEntry(
                coachee_name=self.coachee_name,
                week_number=self.week_number,
                entry_date=self.entry_date,
                central_goal=data.get("central_goal", ""),
                key_results=key_results,
                obstacles=data.get("obstacles", []),
                inspiration_reflection=data.get("inspiration_reflection", ""),
                coach_notes=data.get("coach_notes", ""),
            )
        except Exception as exc:
            logger.warning("LLM extraction failed (%s). Using heuristic fallback.", exc)
            return self._heuristic_journal()

    @staticmethod
    def _parse_kr(text: str, index: int) -> dict:
        """
        Best-effort parse of a KR from free text.

        Looks for a percentage in the text; everything else becomes the name.
        Example: "Improve NPS score from 42 to 60 — 35%" → name=..., pct=35
        """
        import re

        # Use the LAST percentage in the string as the progress value
        # (e.g. "Migrate 50% of accounts — 60%" → progress=60, not 50)
        all_pcts = re.findall(r"(\d{1,3})\s*%", text)
        pct = int(all_pcts[-1]) if all_pcts else 0
        # Remove only the last percentage token from the name
        name = re.sub(r"\s*—?\s*\d{1,3}\s*%\s*$", "", text).strip(" —-,.")
        name = name or f"KR{index + 1}"
        return {"name": name, "target": "", "progress_pct": min(pct, 100), "notes": ""}
