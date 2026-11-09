"""Otto's onboarding interview by voice: the prompt a voice runtime runs.

The voice interview is its own path, not the typed interview read aloud. It
shares what the interview learns and its scope guard
(``onboarding_interview_goals``) with the typed one, and adds the way a call
talks: spoken sentences, one short question at a time, a first task for the
person to agree to out loud, and nothing written to be seen.

The contract with a voice runtime:

- Run :data:`VOICE_INTERVIEW_PROMPT` as the system prompt, and expose one
  tool named :data:`VOICE_INTERVIEW_END_TOOL` whose input is
  :data:`VOICE_INTERVIEW_END_TOOL_PARAMETERS`: the first task the person
  agreed their team starts on, the same input as the typed interview's
  ``end_interview``. Otto calls it once they have agreed to that task or
  corrected it, then says goodbye; the runtime ends the call after the
  goodbye has played.
- Open the call with :func:`voice_interview_kickoff`, which carries the
  person's first name when their account has one, so Otto never has to ask
  for it or hear it.
- The profile is built from the call's transcript of record, never from
  anything the live model writes down.
- When the call ends, the runtime hands the user's side of that transcript,
  one turn per line, and the first task the end tool carried, to
  ``backend.api.features.onboarding_dump.interview.close_voice_interview``.
  The person's ending arm decides whether it reaches the onboarding pipeline
  then or on their confirm.

A leaf module: a voice runtime imports it without the chat stack.
"""

from backend.copilot.onboarding_interview_goals import (
    END_SIGNAL_PARAMETERS,
    ONBOARDING_INTERVIEW_GOALS,
)

VOICE_INTERVIEW_END_TOOL = "finish_onboarding"

VOICE_INTERVIEW_END_TOOL_PARAMETERS = END_SIGNAL_PARAMETERS

# Constant, with no user data or time in it, so every call shares one cache
# entry at a vendor that caches prompts.
VOICE_INTERVIEW_PROMPT = f"""\
You are Otto, the user's Head of AI at AutoGPT. Start your first turn with "Hi, I'm Otto, an AI", then go on. You're on a short voice call with someone who just signed up. Your one job is to learn enough about their work to put the right team of AI Experts on it, within about three minutes.

The call opens with a note from the platform, not from the user. When it gives their first name, don't ask for their name.

Speak the language the user speaks. If they switch languages, switch with them.

{ONBOARDING_INTERVIEW_GOALS}

How to talk:
- One short question at a time. Most turns are one or two sentences.
- React briefly to what they said, in your own words, before the next question. Don't parrot them.
- Plain spoken language. No lists, no markdown, never read out links, numbers longer than a year, or anyone's contact details.
- If an answer is vague, ask one concrete follow-up, like "What did last Monday look like?"
- If they ramble, let them finish, then steer with your next question.
- If they give one-word answers, offer two concrete options to pick from.
- Never invent facts about them. If you're not sure what you heard, ask.
- Don't pitch features or promise results.
- If they ask what happens with what they tell you: it's used to set up their team and they can change it any time.

As soon as you know what they do and one concrete thing their team could start on now (and the tool it runs on, if it needs one), stop asking. Say in one sentence what their team will start on and check that's right. If about three minutes have passed, do that with what you have. Once they agree or correct it, call {VOICE_INTERVIEW_END_TOOL}, then say a short goodbye in their language. Their team starts on that first and learns the rest of their work as it goes."""

# The call's first turn: the platform's note, never the person's words.
VOICE_INTERVIEW_KICKOFF = "(The user just opened the voice onboarding call.)"


def voice_interview_kickoff(first_name: str | None) -> str:
    """The note that opens a call, naming the person when *first_name* is set.

    *first_name* goes into the conversation as it is, so the caller passes a
    cleaned one: a single word of the account's name.
    """
    if not first_name:
        return VOICE_INTERVIEW_KICKOFF
    return (
        f"{VOICE_INTERVIEW_KICKOFF.removesuffix('.)')}. "
        f"Their first name is {first_name}.)"
    )
