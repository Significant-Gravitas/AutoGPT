"""
ChangeNavigatorAgent — Auto-GPT-compatible agent wrapper for the
Change Navigator coaching program.

This agent integrates with Auto-GPT's existing Agent infrastructure while
providing the specialised weekly check-in workflow defined in
autogpt.change_navigator.workflow.

Quick start
-----------
    from autogpt.change_navigator import ChangeNavigatorAgent
    agent = ChangeNavigatorAgent.from_config("change_navigator_settings.yaml")
    agent.run_checkin()
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml

from autogpt.change_navigator.journal import JournalEntry
from autogpt.change_navigator.prompts import SYSTEM_PROMPT
from autogpt.change_navigator.workflow import CheckInWorkflow

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "change_navigator_settings.yaml"


@dataclass
class ChangeNavigatorConfig:
    """Runtime configuration for a single coachee's Change Navigator session."""

    coachee_name: str
    week_number: int
    coach_name: str = "Adi Ben-Nesher"
    language: str = "auto"          # "auto" | "en" | "he"
    llm_model: str = "gpt-4"
    output_dir: str = "change_navigator_output"
    api_budget: float = 0.50        # USD cap per check-in session

    @classmethod
    def from_yaml(cls, path: str | Path = _DEFAULT_CONFIG) -> "ChangeNavigatorConfig":
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str | Path = _DEFAULT_CONFIG) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(
                {k: getattr(self, k) for k in self.__dataclass_fields__},
                fh,
                allow_unicode=True,
            )


class ChangeNavigatorAgent:
    """
    High-level agent that orchestrates the weekly Check-in workflow and
    optionally persists the resulting JournalEntry.

    Parameters
    ----------
    config : ChangeNavigatorConfig
        Session configuration (coachee, week, model, etc.)
    llm : callable, optional
        A function ``(prompt: str) -> str`` for LLM calls.
        When None the agent uses Auto-GPT's ``create_chat_completion``.
    input_fn : callable, optional
        A function ``(prompt: str) -> str`` that reads user input.
        Defaults to ``input()``.
    output_fn : callable, optional
        A function ``(text: str) -> None`` that displays agent messages.
        Defaults to ``print()``.
    """

    SYSTEM_PROMPT = SYSTEM_PROMPT

    def __init__(
        self,
        config: ChangeNavigatorConfig,
        llm: Optional[Callable[[str], str]] = None,
        input_fn: Optional[Callable[[str], str]] = None,
        output_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config
        self._llm = llm or self._default_llm()
        self._input = input_fn or input
        self._output = output_fn or print

    # ---------------------------------------------------------------------------
    # Factory
    # ---------------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        path: str | Path = _DEFAULT_CONFIG,
        **kwargs,
    ) -> "ChangeNavigatorAgent":
        config = ChangeNavigatorConfig.from_yaml(path)
        return cls(config, **kwargs)

    # ---------------------------------------------------------------------------
    # Main entry point
    # ---------------------------------------------------------------------------

    def run_checkin(self) -> Optional[JournalEntry]:
        """
        Run a complete weekly check-in session interactively.

        Returns the final JournalEntry (status=APPROVED) or None if the
        session was interrupted.
        """
        self._output(self.SYSTEM_PROMPT)
        self._output(
            f"\n--- Change Navigator | {self.config.coachee_name} | "
            f"Week {self.config.week_number} ---\n"
        )

        workflow = CheckInWorkflow(
            coachee_name=self.config.coachee_name,
            week_number=self.config.week_number,
        )
        workflow.set_llm(self._llm)

        # Opening question
        opening = workflow.start()
        self._output(f"\nAI Co-Navigator: {opening}")

        while not workflow.is_done():
            try:
                user_input = self._input("\nYou: ")
            except (KeyboardInterrupt, EOFError):
                self._output("\n[Session interrupted]")
                return None

            reply = workflow.advance(user_input)
            self._output(f"\nAI Co-Navigator: {reply}")

        entry = workflow.journal_entry
        if entry:
            self._persist_entry(entry)
        return entry

    # ---------------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------------

    def _persist_entry(self, entry: JournalEntry) -> None:
        """Save the approved journal entry as JSON to the output directory."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{entry.coachee_name.replace(' ', '_')}_"
            f"week{entry.week_number:02d}_"
            f"{entry.entry_date.isoformat()}.json"
        )
        out_path = out_dir / filename
        out_path.write_text(entry.to_json(), encoding="utf-8")
        logger.info("Journal entry saved → %s", out_path)
        self._output(f"\n[Journal saved to {out_path}]")

    # ---------------------------------------------------------------------------
    # LLM integration
    # ---------------------------------------------------------------------------

    def _default_llm(self) -> Callable[[str], str]:
        """
        Return a callable that wraps Auto-GPT's LLM utility.
        Imports are deferred so the module can be imported without a live API key.
        """

        def _call(prompt: str) -> str:
            try:
                from autogpt.llm_utils import create_chat_completion

                messages = [{"role": "user", "content": prompt}]
                return create_chat_completion(
                    messages,
                    model=self.config.llm_model,
                )
            except Exception as exc:
                logger.warning("LLM call failed: %s", exc)
                return "{}"

        return _call

    # ---------------------------------------------------------------------------
    # Auto-GPT AIConfig integration
    # ---------------------------------------------------------------------------

    def to_ai_config(self):
        """
        Return an Auto-GPT AIConfig object pre-configured for this agent.

        This allows the Change Navigator to be launched via the standard
        Auto-GPT CLI (``python -m autogpt --ai-settings ...``).
        """
        from autogpt.config.ai_config import AIConfig

        cfg = AIConfig(
            ai_name="Change Navigator AI",
            ai_role=self.SYSTEM_PROMPT,
            ai_goals=[
                f"Conduct weekly check-in for {self.config.coachee_name}, "
                f"week {self.config.week_number}.",
                "Fill in the weekly journal based on the coachee's answers.",
                "Identify blockers and risks for the next coaching session.",
                "Obtain the coachee's approval before finalising the journal.",
            ],
            api_budget=self.config.api_budget,
        )
        return cfg
