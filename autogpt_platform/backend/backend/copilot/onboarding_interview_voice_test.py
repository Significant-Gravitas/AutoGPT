"""The voice interview's prompt: the shared goals, in a call's own style."""

import ast
from pathlib import Path

from backend.copilot import onboarding_interview_goals, onboarding_interview_voice
from backend.copilot.onboarding_interview_goals import (
    END_SIGNAL_PARAMETERS,
    ONBOARDING_INTERVIEW_GOALS,
)
from backend.copilot.onboarding_interview_voice import (
    VOICE_INTERVIEW_END_TOOL,
    VOICE_INTERVIEW_END_TOOL_PARAMETERS,
    VOICE_INTERVIEW_KICKOFF,
    VOICE_INTERVIEW_PROMPT,
    voice_interview_kickoff,
)


def test_the_call_learns_the_shared_goals_and_holds_the_scope_line() -> None:
    assert ONBOARDING_INTERVIEW_GOALS in VOICE_INTERVIEW_PROMPT
    for goal in (
        "1. What they do: their role and what the business or project is.",
        "2. Who they work with or for: team size and customers.",
        "3. Where their week goes: the recurring work that eats their time.",
        "4. What they want handed off first.",
        "5. The tools that work runs on",
    ):
        assert goal in ONBOARDING_INTERVIEW_GOALS
    assert "treat it as a task for their team" in ONBOARDING_INTERVIEW_GOALS
    assert "you can't help with that" in ONBOARDING_INTERVIEW_GOALS


def test_the_call_talks_like_a_call() -> None:
    assert "You're on a short voice call" in VOICE_INTERVIEW_PROMPT
    assert "Speak the language the user speaks" in VOICE_INTERVIEW_PROMPT
    assert "One short question at a time" in VOICE_INTERVIEW_PROMPT
    assert "No lists, no markdown" in VOICE_INTERVIEW_PROMPT
    assert "never read out links" in VOICE_INTERVIEW_PROMPT
    # Nothing that only means something on a screen or in a chat.
    for written in ("ask_question", "<interview_state>", "<user_context>"):
        assert written not in VOICE_INTERVIEW_PROMPT


def test_the_call_ends_on_an_agreed_first_task_and_the_end_tool() -> None:
    assert (
        "As soon as you know what they do and one concrete thing their team "
        "could start on now (and the tool it runs on, if it needs one), stop "
        "asking. Say in one sentence what their team will start on and check "
        "that's right."
    ) in VOICE_INTERVIEW_PROMPT
    # A live voice model keeps talking after a tool call, so the call comes
    # first and the goodbye after it, before the runtime hangs up.
    assert (
        f"Once they agree or correct it, call {VOICE_INTERVIEW_END_TOOL}, then "
        "say a short goodbye in their language."
    ) in VOICE_INTERVIEW_PROMPT
    assert (
        "Their team starts on that first and learns the rest of their work as "
        "it goes."
    ) in VOICE_INTERVIEW_PROMPT
    assert "sum up what you heard" not in VOICE_INTERVIEW_PROMPT
    # The goodbye is asked for in the person's language, never quoted in one.
    assert "Great, I'm putting your team together now" not in VOICE_INTERVIEW_PROMPT


def test_a_call_that_runs_long_ends_on_what_otto_has() -> None:
    """A call has no reply cap to stop it, so after about three minutes Otto
    proposes the first task with what it has heard. The typed interview has
    its reply cap instead, so the clock is the call's alone."""
    from backend.copilot.model import ONBOARDING_INTERVIEW_KIND
    from backend.copilot.onboarding_interview import get_onboarding_interview_supplement

    assert (
        "Say in one sentence what their team will start on and check that's "
        "right. If about three minutes have passed, do that with what you have. "
        f"Once they agree or correct it, call {VOICE_INTERVIEW_END_TOOL}"
    ) in VOICE_INTERVIEW_PROMPT
    typed = get_onboarding_interview_supplement(ONBOARDING_INTERVIEW_KIND)
    assert "three minutes" not in typed


def test_the_end_tool_takes_what_the_typed_end_signal_takes() -> None:
    """Both signals carry the task the person agreed their team starts on,
    which the hand-off starts on."""
    from backend.copilot.tools import TOOL_REGISTRY

    assert VOICE_INTERVIEW_END_TOOL_PARAMETERS == END_SIGNAL_PARAMETERS
    assert TOOL_REGISTRY["end_interview"].parameters == END_SIGNAL_PARAMETERS
    assert END_SIGNAL_PARAMETERS["required"] == ["first_task"]
    properties = END_SIGNAL_PARAMETERS["properties"]
    assert isinstance(properties, dict)
    assert properties["first_task"] == {
        "type": "string",
        "description": (
            "The first thing their team will start on, as they agreed it "
            "(with any correction they made), in a few words, in their "
            "language."
        ),
    }


def test_no_placeholder_is_left_in_the_prompt() -> None:
    assert "{" not in VOICE_INTERVIEW_PROMPT and "}" not in VOICE_INTERVIEW_PROMPT


def test_the_kickoff_gives_otto_the_name_so_it_never_asks() -> None:
    """A name heard over a call comes back misspelled; one read from the
    account does not, so the kickoff carries it and the prompt says not to
    ask for it."""
    assert voice_interview_kickoff("Dana") == (
        "(The user just opened the voice onboarding call. " "Their first name is Dana.)"
    )
    assert (
        "The call opens with a note from the platform, not from the user. "
        "When it gives their first name, don't ask for their name."
    ) in VOICE_INTERVIEW_PROMPT


def test_a_call_with_no_name_on_the_account_opens_with_the_plain_note() -> None:
    assert voice_interview_kickoff(None) == VOICE_INTERVIEW_KICKOFF
    assert voice_interview_kickoff("") == VOICE_INTERVIEW_KICKOFF
    assert VOICE_INTERVIEW_KICKOFF == (
        "(The user just opened the voice onboarding call.)"
    )


def _imports(module: object) -> set[str]:
    source = Path(module.__file__).read_text()  # type: ignore[attr-defined]
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_a_voice_runtime_imports_the_prompt_without_the_chat_stack() -> None:
    assert _imports(onboarding_interview_voice) == {
        "backend.copilot.onboarding_interview_goals"
    }
    assert _imports(onboarding_interview_goals) == set()
