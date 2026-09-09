"""The team AutoPilot proposes off the back of the brain dump.

Runs as its own background job beside the greeting and the provider
recommendations — same contract as ``recommend.py``: never raises, never
touches the dump status, and always ends with a storable answer.

Two things make this stricter than the provider job. The roster is
content that lags demand (there is no Support or Research template
today), so a need nobody covers has to degrade into a "raise your own"
suggestion rather than a bad hire. And a hire card names a person: the
model only ever picks template ids, and the name, role and avatar on the
card are copied from the template here, so a hallucinated id can never
become a colleague the user is invited to hire.
"""

import asyncio
import logging
import os

from backend.api.features.experts.models import Expert
from backend.api.features.onboarding_dump.models import (
    ExpertRecommendations,
    RaiseSuggestion,
    RecommendedExpert,
)
from backend.api.features.onboarding_dump.parsing import parse_response_json
from backend.util.clients import get_openai_client

logger = logging.getLogger(__name__)

# Same reasoning as the provider job: matching needs against a roster is
# a matching task, and the onboarding loading screen waits on it.
_MODEL = os.environ.get("BRAIN_DUMP_RECOMMEND_MODEL", "anthropic/claude-haiku-4-5")
_TIMEOUT_SECONDS = 30

MAX_EXPERTS = 16
MAX_REASON_CHARS = 140
MAX_DIAGNOSIS_CHARS = 240
MAX_WORKFLOW_NAMES = 3

# The ``/raise`` wizard's role ids. A suggestion is a prefilled link into
# that wizard, so anything outside this set would land on a blank draft.
RAISE_ROLES = frozenset(
    {
        "marketer",
        "sales",
        "developer",
        "researcher",
        "writer",
        "analyst",
        "recruiter",
        "support",
        "operations",
    }
)

# Said without having read anything: the fallback runs when there is no
# transcript, or when the model that would have read it failed.
FALLBACK_DIAGNOSIS = "Here's a first team based on your role and what slows you down."

_PROMPT = """You are AutoPilot, this user's built-in Head of AI. They \
just recorded a brain dump about their work, and your job is to come back \
like a consultant would: name the problems you heard, then propose the \
first hires that take those problems off their plate.

Below is their transcript, what they told the signup wizard, and the \
experts you can actually hire for them today.

Return ONLY valid JSON with exactly these keys:
- "diagnosis": 1-2 sentences, max {max_diagnosis} characters, second \
person, naming the specific problems you heard ("you have a marketing \
problem and a support problem"). Never name an expert here.
- "experts": an array of at most {max_experts} objects, most useful \
first, each with:
  - "template_id": an id copied EXACTLY from the roster below
  - "reason": one sentence, max {max_reason} characters, second person, \
tying this expert to something they actually said
  - "workflow_names": the workflows from THAT expert's list that match \
their work, copied exactly; an empty array is fine
- "raise_suggestion": either null, or an object with "role" and "reason" \
— use it only for a real need that NO expert on the roster covers, and \
only with a role from this list: {raise_roles}. The "reason" is one \
sentence, max {max_reason} characters, second person.

Rules: never invent a template id or a workflow name that is not below; \
never recommend an expert the transcript gives no evidence for (fewer, \
better hires beat filling every available slot); do not promise anything the \
listed workflows cannot do.

Roster:
{roster}

Signup answers:
{answers}

Transcript:
{transcript}
"""


async def generate_expert_recommendations(
    transcript: str,
    *,
    user_role: str | None,
    pain_points: list[str],
    templates: list[Expert],
) -> ExpertRecommendations:
    """Return the team to propose for ``transcript``.

    Never raises. Every failure path — no transcript, no client, a dead
    or malformed generation, a generation with nothing usable left after
    filtering — resolves to :func:`fallback_expert_recommendations`, so
    the greeting page always has a team to render and the client always
    stops polling.
    """
    text = transcript.strip()
    fallback = fallback_expert_recommendations(user_role, pain_points, templates)
    if not text:
        return fallback

    client = get_openai_client(prefer_openrouter=True)
    if client is None:
        logger.warning("Brain dump team: no LLM client configured")
        return fallback

    prompt = _PROMPT.format(
        max_diagnosis=MAX_DIAGNOSIS_CHARS,
        max_experts=MAX_EXPERTS,
        max_reason=MAX_REASON_CHARS,
        raise_roles=", ".join(sorted(RAISE_ROLES)),
        roster=roster_lines(templates),
        answers=_answer_lines(user_role, pain_points),
        transcript=text,
    )
    data = None
    # One retry, for the same reason the greeting has one: a truncated or
    # non-JSON generation is cheaper to redo than to ship around.
    for attempt in range(2):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=1200,
                ),
                timeout=_TIMEOUT_SECONDS,
            )
            data = parse_response_json(response.choices[0].message.content or "")
        except Exception as e:  # degrades to the fallback below
            logger.warning(
                "Brain dump team generation failed (attempt %s): %s", attempt + 1, e
            )
            continue
        if data is not None:
            break
        logger.warning("Brain dump team: non-JSON output (attempt %s)", attempt + 1)
    if data is None:
        return fallback

    team = _parse_team(data, templates)
    if not team.experts and team.raise_suggestion is None:
        # Nothing survived filtering, so there is no door to open. The
        # deterministic mapping is a worse answer than a good generation
        # and a better one than an empty section.
        return fallback
    return team


def roster_lines(templates: list[Expert]) -> str:
    """The hireable roster as one line per template."""
    return "\n".join(
        f"- {template.id}: {template.name} — {template.role}. "
        f"{(template.tagline or '').strip()}. "
        f"Workflows: {', '.join(_template_workflow_names(template))}"
        for template in templates
    )


def _answer_lines(user_role: str | None, pain_points: list[str]) -> str:
    lines = [f"- Role: {user_role}"] if user_role else []
    if pain_points:
        lines.append(f"- Slows them down: {', '.join(pain_points)}")
    return "\n".join(lines) or "- (none given)"


def _template_workflow_names(template: Expert) -> list[str]:
    return [
        workflow.name.strip()
        for workflow in template.workflows
        if workflow.name and workflow.name.strip()
    ]


def _parse_team(data: dict, templates: list[Expert]) -> ExpertRecommendations:
    by_id = {template.id: template for template in templates}
    items = data.get("experts")
    experts: list[RecommendedExpert] = []
    seen: set[str] = set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("template_id"), str):
            continue
        template = by_id.get(item["template_id"].strip())
        if template is None or template.id in seen:
            continue
        seen.add(template.id)
        experts.append(
            _from_template(
                template,
                reason=item.get("reason"),
                workflow_names=item.get("workflow_names"),
            )
        )
        if len(experts) == MAX_EXPERTS:
            break

    diagnosis = data.get("diagnosis")
    return ExpertRecommendations(
        diagnosis=(
            diagnosis.strip()[:MAX_DIAGNOSIS_CHARS]
            if isinstance(diagnosis, str)
            else ""
        ),
        experts=experts,
        raise_suggestion=_parse_raise_suggestion(data.get("raise_suggestion")),
        source="llm",
    )


def _from_template(
    template: Expert, *, reason: object, workflow_names: object
) -> RecommendedExpert:
    return RecommendedExpert(
        template_id=template.id,
        name=template.name,
        role=template.role,
        avatar_url=template.avatar_url,
        reason=reason.strip()[:MAX_REASON_CHARS] if isinstance(reason, str) else "",
        workflow_names=_matched_workflow_names(template, workflow_names),
    )


def _matched_workflow_names(template: Expert, requested: object) -> list[str]:
    """The requested names that are really this template's, its spelling.

    Matching is case-insensitive because the model retypes the names, but
    the chips on the card have to read exactly as the workflow does.
    """
    real = {name.lower(): name for name in _template_workflow_names(template)}
    matched: list[str] = []
    for name in requested if isinstance(requested, list) else []:
        if not isinstance(name, str):
            continue
        actual = real.get(name.strip().lower())
        if actual is not None and actual not in matched:
            matched.append(actual)
        if len(matched) == MAX_WORKFLOW_NAMES:
            break
    return matched


def _parse_raise_suggestion(raw: object) -> RaiseSuggestion | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("role"), str):
        return None
    role = raw["role"].strip().lower()
    if role not in RAISE_ROLES:
        return None
    reason = raw.get("reason")
    return RaiseSuggestion(
        role=role,
        reason=reason.strip()[:MAX_REASON_CHARS] if isinstance(reason, str) else "",
    )


# The wizard's role ids, mapped to the template *roles* that cover them —
# roles rather than template ids or names so a renamed or re-seeded
# template still matches, and a missing one is simply skipped.
_ROLE_TEMPLATE_ROLES: dict[str, tuple[str, ...]] = {
    "Founder/CEO": ("sales", "marketing"),
    "Marketing": ("marketing",),
    "Sales/BD": ("sales",),
    "Operations": ("ops", "office ops", "executive assistant"),
    "Product/PM": ("project management", "ops"),
    "Engineering": ("engineering", "code quality"),
    "HR/People": ("recruiting", "talent sourcing", "ops"),
}
_PAIN_TEMPLATE_ROLES: dict[str, tuple[str, ...]] = {
    "Social media": ("marketing",),
    "Finding leads": ("sales",),
    "Email & outreach": ("sales",),
    "Reports & data": ("ops",),
    "Scheduling": ("ops",),
    "CRM & data entry": ("ops",),
}

# Needs the roster may have no template for — the honest answer is the
# raise door, not the nearest expert. A raise is only suggested when none
# of the template roles that cover it made the team (_RAISE_COVERED_BY),
# so an environment seeded with the engineering or people rosters never
# claims nobody covers a role it just recommended.
_ROLE_RAISE_ROLES: dict[str, str] = {
    "Engineering": "developer",
    "HR/People": "recruiter",
}
_PAIN_RAISE_ROLES: dict[str, str] = {
    "Customer support": "support",
    "Research": "researcher",
}
_RAISE_COVERED_BY: dict[str, tuple[str, ...]] = {
    "developer": ("engineering", "code quality"),
    "recruiter": ("recruiting", "talent sourcing"),
}


def fallback_expert_recommendations(
    user_role: str | None,
    pain_points: list[str],
    templates: list[Expert],
) -> ExpertRecommendations:
    """A team derived from the signup wizard's answers alone.

    Used wherever the model's read of the transcript is unavailable: a
    skipped dump, a failed generation, a process that died mid-job. It
    claims nothing about what the user said, because in those cases we
    may not have heard anything at all.
    """
    wanted = [
        *_ROLE_TEMPLATE_ROLES.get(user_role or "", ()),
        *(
            role
            for point in pain_points
            for role in _PAIN_TEMPLATE_ROLES.get(point, ())
        ),
    ]
    prioritized = [
        template
        for role in dict.fromkeys(wanted)
        for template in templates
        if template.role.strip().lower() == role
    ]
    picked: list[Expert] = []
    seen: set[str] = set()
    for template in [*prioritized, *templates]:
        if template.id not in seen:
            picked.append(template)
            seen.add(template.id)
        if len(picked) == MAX_EXPERTS:
            break

    picked_roles = {template.role.strip().lower() for template in picked}
    return ExpertRecommendations(
        diagnosis=FALLBACK_DIAGNOSIS,
        experts=[_fallback_expert(template) for template in picked],
        raise_suggestion=_fallback_raise(user_role, pain_points, picked_roles),
        source="fallback",
    )


def _fallback_expert(template: Expert) -> RecommendedExpert:
    # The tagline already says what the expert does; without one, fall back
    # to the role so the card never reads blank.
    tagline = (template.tagline or "").strip()
    reason = tagline or f"Covers {template.role.strip().lower()} for you."
    return RecommendedExpert(
        template_id=template.id,
        name=template.name,
        role=template.role,
        avatar_url=template.avatar_url,
        reason=reason[:MAX_REASON_CHARS],
        workflow_names=_template_workflow_names(template)[:MAX_WORKFLOW_NAMES],
    )


def _fallback_raise(
    user_role: str | None, pain_points: list[str], picked_roles: set[str]
) -> RaiseSuggestion | None:
    role = _ROLE_RAISE_ROLES.get(user_role or "") or next(
        (
            _PAIN_RAISE_ROLES[point]
            for point in pain_points
            if point in _PAIN_RAISE_ROLES
        ),
        None,
    )
    if role is None or picked_roles & set(_RAISE_COVERED_BY.get(role, ())):
        return None
    return RaiseSuggestion(
        role=role,
        reason=f"Nobody on the roster covers this yet — you can raise a {role} of your own.",
    )
