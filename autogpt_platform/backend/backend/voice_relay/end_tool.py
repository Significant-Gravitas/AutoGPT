"""The end tool as a request the relay decides on, and the line each tool call
logs.

Otto ends a call by calling the end tool once the person has agreed to the
first task he proposed, and the relay ends the call after the goodbye that
follows. A live model can call the tool with nothing said since its
question, and the tool's answer starts its next turn: a goodbye to someone
who never answered. So the relay accepts the call only when, since Otto's
last completed turn finished playing, its voice detector has heard at least
``MIN_ANSWER_SPEECH_S`` of the person's speech and a caption of theirs has
arrived (:class:`Answer`). Otto's last completed turn is the latest whose
end has arrived; the turn he is in never moves it. In that open turn he may
have acknowledged the answer before calling the tool, and no more: at most
``MAX_ACK_AUDIO_S`` of his audio and no question in his words, or what he
said is a new proposal nobody has answered yet. A call refused in a turn
leaves every later call in that turn refused too. A refused call leaves the
call as it was, and its answer tells Otto to wait for the person and call
again once they have answered.

Each tool call logs one line of counts and names (:class:`ToolCallLine`):
never the first task's text, which is Otto's wording of what the person
said, nor anything said on the call.
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from backend.copilot.onboarding_interview_voice import VOICE_INTERVIEW_END_TOOL

# Speech the person must have said since Otto's last turn played out, beside
# a caption, for the end tool to be accepted.
MIN_ANSWER_SPEECH_S = 0.3
# What Otto may say in his turn before calling the end tool: an
# acknowledgement ("Great!") is short; a proposal is not.
MAX_ACK_AUDIO_S = 4.0

Outcome = Literal["accepted", "refused_unanswered", "repeat", "unknown_tool"]

SAVED: dict[str, object] = {"result": "saved"}
UNKNOWN_TOOL: dict[str, object] = {"error": "unknown tool"}
NOT_CONFIRMED: dict[str, object] = {
    "error": "not_confirmed",
    "message": (
        "The user has not answered yet. Do not say goodbye, and do not end the "
        'call. Say at most "take your time", then wait for them to answer. '
        "Call this tool again only after they have answered out loud."
    ),
}

# What a tool's name and its arguments' names log as.
_UNSAFE = re.compile(r"[^A-Za-z0-9_]")
MAX_NAME_CHARS = 64
MAX_ARG_NAMES = 10


class Answer(BaseModel):
    """What the person has said since Otto's last completed turn finished
    playing: the seconds of speech the voice detector heard, and the
    characters of their captions. It opens once that turn has played out;
    before Otto's first turn has, nothing counts."""

    opened_at: float | None = None
    speech_s: float = 0.0
    caption_chars: int = 0

    def reopen(self, at: float) -> None:
        """Count from *at* (monotonic), when Otto's turn finishes playing."""
        self.opened_at = at
        self.speech_s = 0.0
        self.caption_chars = 0

    def hear(self, seconds: float, now: float) -> None:
        if self.is_open(now):
            self.speech_s += seconds

    def caption(self, text: str, now: float) -> None:
        heard = text.strip()
        if heard and self.is_open(now):
            self.caption_chars += len(heard)

    def is_open(self, now: float) -> bool:
        return self.opened_at is not None and now >= self.opened_at

    @property
    def answered(self) -> bool:
        return self.speech_s >= MIN_ANSWER_SPEECH_S and self.caption_chars > 0

    @property
    def silent(self) -> bool:
        """No caption, and less speech than an answer takes: a quiet room's
        noise is not the person."""
        return self.caption_chars == 0 and self.speech_s < MIN_ANSWER_SPEECH_S


def judge(
    name: str,
    *,
    finishing: bool,
    answer: Answer,
    turn_audio_s: float,
    turn_text: str,
    refused_this_turn: bool,
) -> Outcome:
    """What becomes of a call of tool *name*. *finishing*: the end tool was
    accepted already. *answer*: what the person said since Otto's last
    completed turn played out. *turn_audio_s* and *turn_text*: what Otto has
    said in the turn he is in. *refused_this_turn*: an end-tool call in that
    turn was refused.

    The end tool is accepted only when the person answered, and Otto has
    said no more in this turn than an acknowledgement (at most
    ``MAX_ACK_AUDIO_S`` of audio, and no question), and no call of it was
    refused earlier in this turn. A call that starts a turn has no words in
    it, so the answer alone decides."""
    if name != VOICE_INTERVIEW_END_TOOL:
        return "unknown_tool"
    if finishing:
        return "repeat"
    if (
        not answer.answered
        or refused_this_turn
        or turn_audio_s > MAX_ACK_AUDIO_S
        or "?" in turn_text
    ):
        return "refused_unanswered"
    return "accepted"


def response_for(outcome: Outcome) -> dict[str, object]:
    """The tool's answer to the model for *outcome*."""
    if outcome == "unknown_tool":
        return UNKNOWN_TOOL
    if outcome == "refused_unanswered":
        return NOT_CONFIRMED
    return SAVED


def safe_name(name: str) -> str:
    """*name* as it may be logged: letters, digits and underscores, cut to
    ``MAX_NAME_CHARS``."""
    return _UNSAFE.sub("", name)[:MAX_NAME_CHARS]


class ToolCallLine(BaseModel):
    """The counts and names one tool call logs, and nothing said."""

    model_config = ConfigDict(frozen=True)

    call_id: str
    # The tool's name, made safe (:func:`safe_name`), and which call of it in
    # the call this is, counted from 1.
    name: str
    nth: int
    outcome: Outcome
    # Otto's completed turns, whether he had spoken in the turn under way, and
    # how many seconds of audio that turn held so far.
    turns: int
    otto_speaking: bool
    turn_audio_s: float = 0.0
    # Since Otto's last turn finished playing: how long ago, or None before
    # one has; the person's speech; their captions' characters.
    since_played_s: float | None
    speech_s: float
    caption_chars: int
    # The first task's length, or None when none was given. The end tool's
    # only.
    first_task_chars: int | None = None
    # An unknown tool's argument names, made safe.
    arg_names: tuple[str, ...] = ()

    def describe(self, *, reached: bool) -> str:
        since = (
            "no turn of his played out yet"
            if self.since_played_s is None
            else f"{self.since_played_s:.1f}s since his last turn played"
        )
        said = [
            f"turn {self.turns}{', speaking' if self.otto_speaking else ''}, "
            f"{self.turn_audio_s:.1f}s of his audio in it",
            since,
            f"{self.speech_s:.2f}s of speech and {self.caption_chars} caption "
            f"characters since",
        ]
        if self.name == VOICE_INTERVIEW_END_TOOL:
            said.append(
                "no first task"
                if self.first_task_chars is None
                else f"a first task of {self.first_task_chars} characters"
            )
        if self.outcome == "unknown_tool":
            said.append(f"arguments: {', '.join(self.arg_names) or 'none'}")
        said.append(f"answer {'sent' if reached else 'not sent'}")
        return (
            f"Voice call {self.call_id}: tool {self.name or '?'} call {self.nth} "
            f"{self.outcome} ({'; '.join(said)})"
        )


def arg_names(args: dict[str, object]) -> tuple[str, ...]:
    """An unknown tool's argument names as they may be logged; never their
    values."""
    return tuple(sorted(safe_name(str(name)) for name in args)[:MAX_ARG_NAMES])
