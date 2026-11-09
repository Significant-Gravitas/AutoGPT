"""When Otto may ask the user a question: only when blocked, and not often.

Every question names why the work cannot go on without it. A preference
(tone, format, naming) is asked only in Ask First; in Auto and Unsupervised
Otto picks its default and notes it. "Should I go ahead?" questions are
refused unless the choice cannot be undone, since the gate's approval cards
ask for permission where the mode wants it. A chat may only ask so many
times in a row before Otto has to decide on its own. And a question that is
the first thing in its turn is sent back once: looking first (email, files,
tasks, memory) often settles it, or lets it name the real options and the
deadline; asked again, it goes through.
"""

import re
from collections.abc import Sequence

from backend.copilot.autopilot_mode import AutopilotMode
from backend.copilot.model import ChatMessage

from .models import ClarifyingQuestion

# Turns in a row that may end with a question, per approval mode. One round
# in every mode: once the user has answered, Otto goes on with its defaults
# and says what it assumed. Ask First differs in its approval cards, not in
# asking twice.
QUESTION_BUDGET: dict[AutopilotMode, int] = {
    "unsupervised": 1,
    "auto": 1,
    "ask_first": 1,
}

# At the start of any sentence of the question: "Hugo wanted Thursday. Should
# I tell Tobias yes anyway?" is still a permission ask; "Which bill should I
# pay?" is not.
_PERMISSION_CHATTER = re.compile(
    r"(?:^|[.?!:;]\s+)\s*(should i|shall i|do you want me to|would you like me to|"
    r"want me to|can i go ahead|ok(ay)? to (proceed|go ahead)|"
    r"may i proceed|should we proceed|ready for me to)\b",
    re.IGNORECASE,
)


LOOK_FIRST = (
    "Not asked: you have not looked at anything this turn. Read what bears on "
    "it first (their email, files, tasks, memory, and the plans those point "
    "to): it may settle the question, or let you name every real option and "
    "the deadline. If there is nothing to look up, ask again."
)


def refusal(
    questions: Sequence[ClarifyingQuestion],
    mode: AutopilotMode,
    asked_in_a_row: int,
    looked: bool = True,
) -> str | None:
    """Why these questions may not be asked, or None when they may."""
    problems = [p for q in questions if (p := _problem(q, mode))]
    if problems:
        return (
            "Not asked. "
            + " ".join(problems)
            + " Decide these yourself with your default and note each as an "
            "assumption; ask again only about what truly blocks you."
        )
    budget = QUESTION_BUDGET[mode]
    if asked_in_a_row >= budget:
        return (
            f"Not asked: this chat has already ended {asked_in_a_row} turn(s) in a "
            f"row with a question, the most {mode} mode allows. Go on with your "
            "default_guess for each question and say what you assumed."
        )
    if not looked:
        return LOOK_FIRST
    return None


def _problem(question: ClarifyingQuestion, mode: AutopilotMode) -> str | None:
    label = f"'{question.question}':"
    if question.reason is None:
        return f"{label} give the reason it blocks you and a default_guess."
    if question.reason == "preference" and mode != "ask_first":
        return f"{label} a preference is not asked in {mode} mode."
    if (
        _PERMISSION_CHATTER.search(question.question)
        and question.reason != "irreversible_choice"
    ):
        return f"{label} do not ask permission to do the work you were asked for."
    return None


def questions_in_a_row(messages: Sequence[ChatMessage]) -> int:
    """How many turns before the user's latest message ended with a question,
    counting back until a turn that did not."""
    user_turns = [i for i, m in enumerate(messages) if m.role == "user"]
    count = 0
    for start, end in zip(reversed(user_turns[:-1]), reversed(user_turns[1:])):
        if not any(_asks(m) for m in messages[start + 1 : end]):
            break
        count += 1
    return count


def _asks(message: ChatMessage) -> bool:
    return message.role == "assistant" and any(
        (call.get("function") or {}).get("name") == "ask_question"
        for call in message.tool_calls or []
    )
