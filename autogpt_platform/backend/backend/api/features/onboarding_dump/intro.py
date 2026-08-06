"""The copilot home's onboarding greeting: what AutoPilot says it heard.

Generated while the onboarding loading screen is still up, then stored —
landing on ``/copilot`` must never wait on a model. If generation fails the
greeting falls back to a template, so there is always something to show.

The generation instructions live in Langfuse (``BRAIN_DUMP_GREETING_PROMPT_NAME``)
so they can be tuned without a deploy; ``_LOCAL_PROMPT`` below is the
fallback when Langfuse is unconfigured or unreachable.
"""

import asyncio
import json
import logging
import os
import re

from langfuse import get_client

from backend.api.features.onboarding_dump.models import SuggestedPrompt
from backend.util.clients import get_openai_client
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

_MODEL = os.environ.get("BRAIN_DUMP_GREETING_MODEL", "anthropic/claude-sonnet-5")
_TIMEOUT_SECONDS = 30

LANGFUSE_PROMPT_NAME = os.environ.get(
    "BRAIN_DUMP_GREETING_PROMPT_NAME", "Brain Dump Greeting"
)
LANGFUSE_PROMPT_CACHE_TTL_SECONDS = 300

MIN_PROMPTS = 5
MAX_PROMPTS = 6
MAX_GREETING_CHARS = 500
MAX_TITLE_CHARS = 120
MAX_PROMPT_CHARS = 2_000

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

# The local fallback for the Langfuse-managed instructions. The greeting
# reflects the dump back so the user can see they were heard; the suggested
# prompts are what turn that into an action. A Langfuse edit must keep the
# same JSON contract — a malformed generation degrades to the template.
_LOCAL_PROMPT = """You are AutoPilot, an AI teammate that can run real \
recurring automations: watch sources, draft content, send digests, build \
agents that work while the user sleeps. A new user just recorded a short \
spoken brain dump about their work. Write the greeting they will see when \
they first open the app.

Return ONLY valid JSON with exactly these keys:
- "greeting": 2-3 sentences, max 450 characters, second person, warm and \
concrete. Show them you listened by naming the specific things they \
actually said. This is their FIRST time in the app — never say "welcome \
back" or imply any prior visit or conversation. The app already renders \
"Hey, <name>" directly above this text, so NEVER use the user's name or \
any other salutation ("good to have you here", "welcome") — open \
directly with what you heard. Do not ask a question, do not promise \
anything you were not told about, and never mention what you cannot do \
or cannot reach — talk only about what you will take on.
- "prompts": an array of 5-6 objects, each with:
  - "title": 8-14 words. Each title must feel like a small product built \
specifically for THIS person — assembled from the exact tools, projects, \
clients and pains they named. The reaction you are engineering is "wait, \
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
    """
    text = transcript.strip()
    if not text:
        return fallback_intro(text)

    client = get_openai_client(prefer_openrouter=True)
    if client is None:
        logger.warning("Brain dump greeting: no LLM client configured")
        return fallback_intro(text)

    instructions = await _fetch_langfuse_prompt() or _LOCAL_PROMPT
    data = None
    # Two attempts: at temperature 0.6 an occasional generation comes back
    # truncated or malformed, and one retry is far cheaper than shipping
    # the generic fallback to a brand-new user.
    for attempt in range(2):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=_MODEL,
                    messages=[{"role": "user", "content": f"{instructions}{text}"}],
                    temperature=0.6,
                    max_tokens=3000,
                ),
                timeout=_TIMEOUT_SECONDS,
            )
            data = _parse_response_json(response.choices[0].message.content or "")
        except Exception as e:  # degrades to the template below
            logger.warning(
                "Brain dump greeting generation failed (attempt %s): %s",
                attempt + 1,
                e,
            )
            continue
        if data is not None:
            break
        logger.warning("Brain dump greeting: non-JSON output (attempt %s)", attempt + 1)
    if data is None:
        return fallback_intro(text)

    greeting = data.get("greeting")
    if not isinstance(greeting, str) or not greeting.strip():
        return fallback_intro(text)

    prompts = _parse_prompts(data.get("prompts"))
    if len(prompts) < MIN_PROMPTS:
        # A greeting with two suggestions under it looks broken; the
        # generic set is better than a half-empty one.
        return greeting.strip()[:MAX_GREETING_CHARS], fallback_prompts()
    return greeting.strip()[:MAX_GREETING_CHARS], prompts[:MAX_PROMPTS]


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


def _parse_response_json(content: str) -> dict | None:
    """Parse the model's JSON, tolerating markdown fences and preamble.

    Anthropic models have no OpenAI-style JSON mode, so the contract is
    prompt-level ("return ONLY valid JSON") and the parser forgives the
    two ways that commonly bends: a ```json fence around the object, or
    stray prose before/after it.
    """
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _parse_prompts(raw: object) -> list[SuggestedPrompt]:
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
        icon = item.get("icon")
        prompts.append(
            SuggestedPrompt(
                title=title.strip()[:MAX_TITLE_CHARS],
                prompt=prompt.strip()[:MAX_PROMPT_CHARS],
                icon=icon if icon in PROMPT_ICONS else DEFAULT_PROMPT_ICON,
            )
        )
    return prompts


def fallback_intro(transcript: str) -> tuple[str, list[SuggestedPrompt]]:
    """A greeting that is true even when the model gave us nothing.

    Deliberately makes no claim about *what* was said — inventing detail
    here would be worse than being generic.
    """
    if not transcript.strip():
        return (
            "I'm ready when you are. Tell me what your week looks like and "
            "I'll find the parts worth handing over. Here are a few places "
            "we could start.",
            fallback_prompts(),
        )
    return (
        "Thanks for talking me through your work — I've got it. "
        "Here are a few places I can start.",
        fallback_prompts(),
    )


def fallback_prompts() -> list[SuggestedPrompt]:
    """Generic starters used when generation fails or there is no dump.

    Phrased as things the user might genuinely want on day one, not as
    claims about this particular user.
    """
    return [
        SuggestedPrompt(
            title="Build an agent that drafts your Monday status report before you wake up",
            prompt=(
                "I want a recurring weekly summary of my work. Ask me what "
                "sources to pull from, then draft the first report so I can "
                "see the format before it goes on a schedule."
            ),
            icon="calendar-check",
        ),
        SuggestedPrompt(
            title="Get a sourced research brief on any topic in minutes, not afternoons",
            prompt=(
                "I need research support. Let me give you a topic and I want "
                "back a concise brief with sources — start by asking me what "
                "I'm currently trying to learn about."
            ),
            icon="magnifying-glass",
        ),
        SuggestedPrompt(
            title="Put an always-on watch on competitors and get pinged the moment they move",
            prompt=(
                "I want to monitor competitors. Help me pick who to track and "
                "what changes matter — pricing, product launches, messaging — "
                "and set up alerts for them."
            ),
            icon="bell",
        ),
        SuggestedPrompt(
            title="Have your next post or newsletter drafted in your voice, ready to edit",
            prompt=(
                "I regularly need written content. Ask me what I publish and "
                "how often, then draft the next one in my voice so I can see "
                "how close you get."
            ),
            icon="megaphone",
        ),
        SuggestedPrompt(
            title="Find the two tasks eating your week and hand them to an agent today",
            prompt=(
                "Help me find what to automate. Interview me briefly about "
                "what repeats every week, then propose the two tasks you'd "
                "take over first and how."
            ),
            icon="robot",
        ),
    ]
