"""The copilot home's onboarding greeting: what Otto says it heard.

Generated while the onboarding loading screen is still up, then stored —
landing on ``/copilot`` must never wait on a model. If generation fails the
greeting falls back to a template, so there is always something to show.

The generation instructions live in Langfuse (``BRAIN_DUMP_GREETING_PROMPT_NAME``)
so they can be tuned without a deploy; ``_LOCAL_PROMPT`` below is the
fallback when Langfuse is unconfigured or unreachable.
"""

import asyncio
import logging
import os

from langfuse import get_client
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from backend.api.features.onboarding_dump.models import SuggestedPrompt
from backend.api.features.onboarding_dump.parsing import parse_response_json
from backend.api.features.onboarding_dump.providers import connectable_lines
from backend.util.clients import get_openai_client
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

_MODEL = os.environ.get("BRAIN_DUMP_GREETING_MODEL", "anthropic/claude-sonnet-5-5")
_TIMEOUT_SECONDS = 45
# The model thinks adaptively and the thinking counts against this: at 3,000
# one call in five came back cut off mid-JSON.
_MAX_TOKENS = 6_000

# The Langfuse copy of these instructions wins whenever it is reachable,
# so it must be versioned in lockstep with ``_LOCAL_PROMPT`` — a voice
# change made only here is invisible in prod.
LANGFUSE_PROMPT_NAME = os.environ.get(
    "BRAIN_DUMP_GREETING_PROMPT_NAME", "Brain Dump Greeting"
)
LANGFUSE_PROMPT_CACHE_TTL_SECONDS = 300

MIN_PROMPTS = 5
MAX_PROMPTS = 6
# Asked of the model by both copies of the instructions; MAX_GREETING_CHARS
# clips what comes back.
GREETING_CHAR_LIMIT = 600
MAX_GREETING_CHARS = 650
MAX_TITLE_CHARS = 120
MAX_PROMPT_CHARS = 2_000
MAX_INTEGRATIONS = 8
MAX_INTEGRATION_CHARS = 60

# Phosphor icon slugs the model may pick from — the frontend maps these to
# real icon components and falls back to "sparkle" for anything else.
PROMPT_ICONS = frozenset(
    {
        "sparkle",
        "chart-bar",
        "envelope",
        "magnifying-glass",
        "calendar-check",
        "bell",
        "rocket-launch",
        "file-text",
        "globe",
        "code",
        "newspaper",
        "users",
        "shopping-cart",
        "chats",
        "lightning",
        "target",
        "robot",
        "clock",
        "megaphone",
        "currency-dollar",
    }
)
DEFAULT_PROMPT_ICON = "sparkle"

# Sent as the system message on every generation, whichever copy of the
# instructions is in use: the Langfuse copy wins in prod and predates it.
COVERAGE_CONTRACT = f"""Leave out none of the problems the user described. \
The greeting names every one, grouping those that belong together. Name a \
problem no workflow or expert can take over too (a machine, a recipe, \
on-site hardware, a personal decision), as something you will work through \
with them yourself, never as a job for the team. When the transcript has \
such a problem, make the last suggested prompt that work. The greeting may \
run to {GREETING_CHAR_LIMIT} characters when it has to."""

# Sent as a second system message on every generation, for the same reason:
# the Langfuse copy can predate this field, and the Home checklist labels
# each task from it. Code-owned so the label never depends on a prompt
# edit landing.
INTEGRATIONS_CONTRACT = """Every object in "prompts" also has \
"integrations": an array naming each outside service the task signs in to. \
Use the id from the integrations list for a service on it. A service the \
task needs that is not on the list goes in by its own name ("Shopify") and \
the app shows it as not supported yet, so prefer tasks that run entirely on \
listed services, include at most one that needs a service not on the list, \
and put it last. Use [] when the task needs no sign-in."""

# The local fallback for the Langfuse-managed instructions. The greeting
# diagnoses the dump so the user can see they were heard; the suggested
# prompts are what turn that into an action. A Langfuse edit must keep the
# same JSON contract — a malformed generation degrades to the template.
_LOCAL_PROMPT = f"""You are Otto, this user's built-in Head of AI. \
You are theirs alone, never shared, and you run a team for them: you can \
build real recurring automations (watch sources, draft content, send \
digests, run agents while they sleep) and you can bring in experts to own \
whole areas of their work. A new user just recorded a short spoken brain \
dump about their work. Write the greeting they will see when they first \
open the app.

Return ONLY valid JSON with exactly these keys:
- "greeting": 2-3 sentences, max {GREETING_CHAR_LIMIT} characters, second person, warm and \
concrete. Diagnose, the way a consultant would on day one: name the \
specific problems you heard in their own terms ("you have a marketing \
problem and a support problem"), then say you are putting a team \
together to take them on. NEVER name a specific expert, and never say \
anyone has been hired or assigned — the team is proposed separately, \
below your text, and naming someone here would promise a colleague who \
may not exist. This is their FIRST time in the app — never say "welcome \
back" or imply any prior visit or conversation. The app already renders \
"Hey, <name>" directly above this text, so NEVER use the user's name or \
any other salutation ("good to have you here", "welcome") — open \
directly with what you heard. Do not ask a question, do not promise \
anything you were not told about, and never mention what you cannot do \
or cannot reach — talk only about what you will take on.
- "prompts": an array of 5-6 objects, each with:
  - "title": 8-14 words. Each title must feel like a small product built \
specifically for THIS person — assembled from the projects, clients and \
pains they named, and the tools they named that this platform can connect. The reaction you are engineering is "wait, \
I can do THAT here?". Prefer compound automations that chain steps into \
one flow (watch a source → extract what matters → draft the response → \
deliver it on a schedule) over single actions. The test: if a title \
could be shown to a random stranger and still make sense, it is too \
generic — rewrite it until it could only belong to this user. Banned: \
"automate repetitive tasks", "organize your work", "stay on top of", \
"streamline", "boost productivity" and anything equally hollow.
  - "prompt": the full message to send on their behalf if they pick it — \
2-4 sentences, first person as the user, carrying over the concrete \
details from the title so work can start immediately.
  - "icon": the best-fitting slug from exactly this list: sparkle, \
chart-bar, envelope, magnifying-glass, calendar-check, bell, \
rocket-launch, file-text, globe, code, newspaper, users, shopping-cart, \
chats, lightning, target, robot, clock, megaphone, currency-dollar.
  - "integrations": see the system message.

Example of the required jump in specificity (do NOT copy these, derive \
your own from the transcript):
- weak: "Automate your email follow-ups"
- strong: "Chase unpaid Shopify invoices with polite follow-ups that \
escalate every 3 days"
- weak: "Get a summary of your meetings"
- strong: "Turn every sales call into a CRM update and a follow-up \
draft before you're back at your desk"

Order the prompts most-valuable first. If the transcript is thin, fill \
the remaining slots with the most impressive broadly useful automations \
you can offer (a morning digest of their industry's news, an always-on \
competitor monitor, a weekly auto-drafted report) — but never invent \
personal details that are not in the transcript.

Transcript:
"""


async def generate_intro(transcript: str) -> tuple[str, list[SuggestedPrompt]]:
    """Return ``(greeting, prompts)`` for ``transcript``.

    Never raises: a failed or malformed generation degrades to the
    template below rather than costing the user their greeting.

    The greeting is unvalidated model output. ``transcript`` is the user's
    own words, and the "only name real integrations" constraint in the
    prompt is advisory, so an injected or hallucinated tool name can reach
    the prose. That surface is deliberately left open: the text is shown
    only to the person who recorded it, and the ids that become
    connectable or recommended tiles are validated against
    :func:`providers.known_providers` in ``recommend.py`` rather than
    trusted from prose.
    """
    text = transcript.strip()
    if not text:
        return fallback_intro(text)

    client = get_openai_client(prefer_openrouter=True)
    if client is None:
        logger.warning("Brain dump greeting: no LLM client configured")
        return fallback_intro(text)

    instructions = await _fetch_langfuse_prompt() or _LOCAL_PROMPT
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=COVERAGE_CONTRACT),
        ChatCompletionSystemMessageParam(role="system", content=INTEGRATIONS_CONTRACT),
        ChatCompletionUserMessageParam(
            role="user", content=f"{_integrations_block()}{instructions}{text}"
        ),
    ]
    best: tuple[str, list[SuggestedPrompt]] | None = None
    # Two attempts: at temperature 0.6 an occasional generation comes back
    # truncated, malformed, without a greeting or without the integrations
    # of enough prompts, and one retry is far cheaper than shipping the
    # generic fallback to a brand-new user. A retry never replaces a better
    # attempt, and greeting and prompts always come from the same one.
    for attempt in range(2):
        generated = await _generate(client, messages, attempt)
        greeting = _greeting_of(generated)
        if generated is None or greeting is None:
            continue
        parsed = _parse_prompts(generated.get("prompts"))
        if best is None or _rank(parsed) > _rank(best[1]):
            best = (greeting, parsed)
        if len(_declared(best[1])) >= MIN_PROMPTS:
            break
        logger.warning(
            "Brain dump greeting: too few prompts with integrations (attempt %s)",
            attempt + 1,
        )
    if best is None:
        return fallback_intro(text)
    greeting, parsed = best
    return greeting, _usable_prompts(parsed)


def _usable_prompts(parsed: list[SuggestedPrompt]) -> list[SuggestedPrompt]:
    """The prompts to show: the ones that declare their integrations; or,
    when the model left that out of too many, all of its prompts, so a
    model that ignores the field costs the labels' precision rather than
    every new user their own tasks (each label still reads the task's
    text); or the generic set, when there are too few to look whole."""
    declared = _declared(parsed)
    if len(declared) >= MIN_PROMPTS:
        return declared[:MAX_PROMPTS]
    if len(parsed) >= MIN_PROMPTS:
        logger.warning(
            "Brain dump greeting: the model left out integrations on "
            "%s of %s prompts; keeping them, labelled from their text",
            len(parsed) - len(declared),
            len(parsed),
        )
        return parsed[:MAX_PROMPTS]
    # A greeting with two suggestions under it looks broken; the generic
    # set is better than a half-empty one.
    return fallback_prompts()


def _declared(prompts: list[SuggestedPrompt]) -> list[SuggestedPrompt]:
    return [prompt for prompt in prompts if prompt.integrations is not None]


def _rank(prompts: list[SuggestedPrompt]) -> tuple[bool, int, int]:
    """Better attempts first: enough declared prompts, then more of them,
    then more prompts at all."""
    declared = len(_declared(prompts))
    return declared >= MIN_PROMPTS, declared, len(prompts)


def _greeting_of(data: dict | None) -> str | None:
    """The generation's greeting, trimmed, or ``None`` if it has none."""
    greeting = data.get("greeting") if data else None
    if not isinstance(greeting, str) or not greeting.strip():
        return None
    return _clip_greeting(greeting)


async def _generate(
    client: AsyncOpenAI, messages: list[ChatCompletionMessageParam], attempt: int
) -> dict | None:
    """One generation, parsed; ``None`` when it failed or was not JSON."""
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=_MODEL,
                messages=messages,
                temperature=0.6,
                max_tokens=_MAX_TOKENS,
            ),
            timeout=_TIMEOUT_SECONDS,
        )
        data = parse_response_json(response.choices[0].message.content or "")
    except Exception as e:  # degrades to the template
        logger.warning(
            "Brain dump greeting generation failed (attempt %s): %s", attempt + 1, e
        )
        return None
    if data is None:
        logger.warning("Brain dump greeting: non-JSON output (attempt %s)", attempt + 1)
    return data


def _integrations_block() -> str:
    """The live provider registry, prepended to the instructions.

    Without it the model invents the tools it promises to wire up, and the
    user's first suggested automation is one we cannot run. Goes in front
    of the instructions (which end with ``Transcript:``) so the whole
    static half of the message stays a stable prefix.
    """
    lines = connectable_lines()
    if not lines:
        return ""
    return (
        "These are the integrations this platform can connect to. Build "
        "the automations you promise on them: a tool that is not on this "
        "list cannot be connected yet.\n"
        f"{lines}\n\n"
    )


async def _fetch_langfuse_prompt() -> str | None:
    """Fetch the greeting instructions from Langfuse.

    Returns the compiled prompt string, or None when Langfuse is
    unconfigured or the fetch fails — the caller falls back to
    ``_LOCAL_PROMPT`` so a Langfuse outage never blocks onboarding.
    """
    if not (
        settings.secrets.langfuse_public_key and settings.secrets.langfuse_secret_key
    ):
        return None
    try:
        prompt = await asyncio.to_thread(
            get_client().get_prompt,
            LANGFUSE_PROMPT_NAME,
            cache_ttl_seconds=LANGFUSE_PROMPT_CACHE_TTL_SECONDS,
        )
        return prompt.compile()
    except Exception as e:  # local prompt is the fallback
        logger.warning("Brain dump greeting: Langfuse prompt fetch failed: %s", e)
        return None


def _clip_greeting(greeting: str) -> str:
    """``greeting`` within the cap, at its last sentence end if that keeps most
    of it, otherwise at a word boundary."""
    text = greeting.strip()
    if len(text) <= MAX_GREETING_CHARS:
        return text
    window = text[: MAX_GREETING_CHARS + 1]
    end = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if end >= MAX_GREETING_CHARS // 2:
        return text[: end + 1]
    cut = text[:MAX_GREETING_CHARS]
    return cut[: cut.rfind(" ")].rstrip(",;:") + "…"


def _parse_prompts(raw: object) -> list[SuggestedPrompt]:
    """The usable prompts in ``raw``. One that does not say what it
    connects to keeps ``integrations=None`` ("not declared"), which the
    caller prefers to avoid (``_usable_prompts``)."""
    if not isinstance(raw, list):
        return []
    prompts = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        title, prompt = item.get("title"), item.get("prompt")
        if not (isinstance(title, str) and title.strip()):
            continue
        if not (isinstance(prompt, str) and prompt.strip()):
            continue
        integrations = _parse_integrations(item.get("integrations"))
        icon = item.get("icon")
        prompts.append(
            SuggestedPrompt(
                title=title.strip()[:MAX_TITLE_CHARS],
                prompt=prompt.strip()[:MAX_PROMPT_CHARS],
                icon=prompt_icon(icon),
                integrations=integrations,
            )
        )
    return prompts


def prompt_icon(raw: object) -> str:
    """``raw`` if it is one of our icon slugs, else the default. Checks the
    type first: a list or dict is unhashable, and the set lookup would raise
    where a bad icon should only cost its icon."""
    return raw if isinstance(raw, str) and raw in PROMPT_ICONS else DEFAULT_PROMPT_ICON


def _parse_integrations(raw: object) -> list[str] | None:
    """The services a prompt says it connects to; ``None`` if it did not say.

    ``[]`` means no sign-in. A non-empty array with no usable name in it
    (``[7]``) says nothing, not that: it would hide the service it meant.
    """
    if not isinstance(raw, list):
        return None
    names = [
        name.strip()[:MAX_INTEGRATION_CHARS]
        for name in raw
        if isinstance(name, str) and name.strip()
    ]
    if raw and not names:
        return None
    return names[:MAX_INTEGRATIONS]


def fallback_intro(transcript: str) -> tuple[str, list[SuggestedPrompt]]:
    """A greeting that is true even when the model gave us nothing.

    Deliberately makes no claim about *what* was said, and names nobody —
    inventing either would be worse than being generic.
    """
    if not transcript.strip():
        return (
            "I'm your Head of AI here. Tell me what your week looks like "
            "and I'll work out who and what you need. Here are a few "
            "places we could start.",
            fallback_prompts(),
        )
    return (
        "Thanks for talking me through your work — I've got it. I'm your "
        "Head of AI here, and I'll start putting a team around this. "
        "Here are a few places I can start.",
        fallback_prompts(),
    )


def fallback_prompts() -> list[SuggestedPrompt]:
    """Generic starters used when generation fails or there is no dump.

    Phrased as things the user might genuinely want on day one, not as
    claims about this particular user. None of them needs a service, and
    each says so.
    """
    return [
        SuggestedPrompt(
            title="Build a workflow that drafts your Monday status report before you wake up",
            prompt=(
                "I want a recurring weekly summary of my work. Ask me what "
                "sources to pull from, then draft the first report so I can "
                "see the format before it goes on a schedule."
            ),
            icon="calendar-check",
            integrations=[],
        ),
        SuggestedPrompt(
            title="Get a sourced research brief on any topic in minutes, not afternoons",
            prompt=(
                "I need research support. Let me give you a topic and I want "
                "back a concise brief with sources — start by asking me what "
                "I'm currently trying to learn about."
            ),
            icon="magnifying-glass",
            integrations=[],
        ),
        SuggestedPrompt(
            title="Put an always-on watch on competitors and get pinged the moment they move",
            prompt=(
                "I want to monitor competitors. Help me pick who to track and "
                "what changes matter — pricing, product launches, messaging — "
                "and set up alerts for them."
            ),
            icon="bell",
            integrations=[],
        ),
        SuggestedPrompt(
            title="Have your next post or newsletter drafted in your voice, ready to edit",
            prompt=(
                "I regularly need written content. Ask me what I publish and "
                "how often, then draft the next one in my voice so I can see "
                "how close you get."
            ),
            icon="megaphone",
            integrations=[],
        ),
        SuggestedPrompt(
            title="Find the two tasks eating your week and hand them to an expert today",
            prompt=(
                "Help me find what to automate. Interview me briefly about "
                "what repeats every week, then propose the two tasks you'd "
                "take over first and how."
            ),
            icon="robot",
            integrations=[],
        ),
    ]
