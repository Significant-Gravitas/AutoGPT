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
  corrected it out loud, then says goodbye; the runtime ends the call after
  the goodbye has played, and refuses a call made before the person
  answered.
- Open the call with :func:`voice_interview_kickoff`, which carries the
  call's language (:func:`call_language`) and the person's first name when
  their account has one, so Otto never has to ask for it or hear it.
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

# The languages a call can start in, by BCP-47 primary language subtag: ones
# the live model speaks. A call starts in the person's browser language when
# it is one of these, and in English otherwise.
CALL_LANGUAGES: dict[str, str] = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
}
DEFAULT_CALL_LANGUAGE = "en"
# Longer than any language tag a browser sends.
MAX_LANGUAGE_TAG_CHARS = 35

# Constant, with no user data or time in it, so every call shares one cache
# entry at a vendor that caches prompts.
VOICE_INTERVIEW_PROMPT = f"""\
You are Otto, the user's Head of AI at AutoGPT. Start your first turn with "Hi, I'm Otto, an AI", then go on. You're on a short voice call with someone who just signed up. Your one job is to learn enough about their work to put the right team of AI Experts on it, within about three minutes.

The call opens with a note from the platform, not from the user. When it gives their first name, don't ask for their name.

The platform's note says which language the call is in. Speak it and stay in it. Switch only when the user asks you to, or speaks full sentences in another language two turns running; then switch with them. Never switch because of one short or unclear turn: a short turn that seems to be in another language was most likely misheard, so carry on in the call's language.

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

When you name the first task, say in one sentence what their team will start on (and the tool it runs on, if it needs one) and check that's right. If about three minutes have passed, do that with what you have. Then wait for them to answer out loud, and never answer for them; if they say nothing, ask once more. Once they agree or correct it, call {VOICE_INTERVIEW_END_TOOL}, then say a short goodbye in their language. Their team starts on that first and learns the rest of their work as it goes."""

# The call's first turn: the platform's note, never the person's words.
VOICE_INTERVIEW_KICKOFF = "(The user just opened the voice onboarding call.)"


def call_language(tag: object) -> str:
    """The language a call starts in for the BCP-47 tag *tag*, such as a
    browser's ``es-MX``: its primary subtag when it is one of
    :data:`CALL_LANGUAGES`, and :data:`DEFAULT_CALL_LANGUAGE` for anything
    else, including no tag at all."""
    if not isinstance(tag, str) or len(tag) > MAX_LANGUAGE_TAG_CHARS:
        return DEFAULT_CALL_LANGUAGE
    primary = tag.strip().replace("_", "-").split("-", 1)[0].lower()
    return primary if primary in CALL_LANGUAGES else DEFAULT_CALL_LANGUAGE


def language_name(language: str) -> str:
    """The English name of *language*, one of :data:`CALL_LANGUAGES`."""
    return CALL_LANGUAGES[call_language(language)]


def voice_interview_kickoff(
    first_name: str | None, language: str = DEFAULT_CALL_LANGUAGE
) -> str:
    """The note that opens a call in *language*, naming the person when
    *first_name* is set.

    *first_name* goes into the conversation as it is, so the caller passes a
    cleaned one: a single word of the account's name.
    """
    note = (
        f"{VOICE_INTERVIEW_KICKOFF.removesuffix('.)')}. "
        f"The call is in {language_name(language)}."
    )
    if first_name:
        note += f" Their first name is {first_name}."
    return f"{note})"
