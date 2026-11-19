"""Next tasks Otto drafts for the user, ready with the morning brief.

One cheap model call reads the user's recent tasks and their results, the
expert roster, and what is already suggested or was dismissed, and drafts up
to three tasks. Each is stored as a ``Task`` with ``isSuggested`` set, which
keeps it off the board until the user starts it.

The rules hold in code as well as in the prompt: a draft restating a board
task, a live, dismissed or recently retired suggestion, or another draft, even
in other words (``repeats.py``), is dropped; at most
``MAX_SUGGESTIONS`` stay alive, and ones older than ``STALE_AFTER`` are
retired so newer drafts can take their place; an account with no tasks gets
none. A failed call drafts nothing and the caller carries on.
"""

import asyncio
import logging
import re
import uuid
from datetime import UTC, datetime, timedelta

from backend.copilot.config import ChatConfig
from backend.copilot.inference.complete import StructuredCompletion, structured_complete
from backend.copilot.inference.context import (
    InferenceContext,
    InferenceError,
    InferenceJob,
    InferenceScope,
    InferenceUsage,
)
from backend.copilot.inference.record import record
from backend.copilot.inference.routing import resolve_route
from backend.copilot.trial_cost_context import attributed_usage
from backend.copilot.usage_activation import UsageActivationUnavailable
from backend.data.db_accessors import experts_db, task_db, task_suggestion_db
from backend.data.task import Task
from backend.data.task_suggestion import SuggestedTaskInput

from .repeats import TakenItem, is_repeat
from .suggestion_prompt import (
    MAX_SUGGESTIONS,
    SuggestionDraft,
    SuggestionsResponse,
    facts_block,
    system_prompt,
)

logger = logging.getLogger(__name__)

config = ChatConfig()

STALE_AFTER = timedelta(days=3)
_BOARD_LIMIT = 100
_TIMEOUT_SECONDS = 20.0
_TOTAL_BUDGET_SECONDS = 30.0
_MIN_ATTEMPT_SECONDS = 3.0
_ATTEMPTS = 2
# Three drafts of a title, a goal, a done-when, a short brief and a reason,
# plus the JSON.
_MAX_OUTPUT_TOKENS = 1400
# Two titles whose word sets overlap this much are the same task reworded.
_SAME_TITLE_OVERLAP = 0.8
_WORD_RE = re.compile(r"[a-z0-9]+")


async def draft_task_suggestions(
    user_id: str, now: datetime | None = None
) -> list[Task]:
    """Top the user's live suggestions back up to ``MAX_SUGGESTIONS`` and
    return them, newest first. Database errors raise; a failed model call
    only means nothing new was drafted."""
    now = now or datetime.now(UTC)
    await task_suggestion_db().retire_stale_suggestions(user_id, now - STALE_AFTER)
    board, live, dismissed, experts = await asyncio.gather(
        task_db().list_tasks(
            user_id=user_id, limit=_BOARD_LIMIT, include_subtasks=False
        ),
        task_db().list_tasks(user_id=user_id, suggested=True, limit=MAX_SUGGESTIONS),
        # Dismissed ones, and ones retired unstarted lately (the retire above
        # included): neither comes back.
        task_suggestion_db().dismissed_suggestion_titles(user_id),
        experts_db().list_experts(user_id),
    )
    room = MAX_SUGGESTIONS - len(live)
    if room <= 0 or not board:
        return live
    facts = facts_block(
        board=board,
        suggested=live,
        set_aside_titles=dismissed,
        experts=experts,
        now=now,
    )
    try:
        response = await _ask_model(user_id, room, facts)
    except UsageActivationUnavailable:
        logger.warning("Task suggestions usage attribution unavailable")
        response = None
    if response is None:
        return live
    drafts = keep_new_drafts(
        response.suggestions,
        taken=[TakenItem(title=t.title, goal=t.goal) for t in [*board, *live]]
        + [TakenItem(title=title) for title in dismissed],
        expert_ids={expert.id for expert in experts},
        room=room,
    )
    created = [
        await task_suggestion_db().create_suggested_task(user_id, draft)
        for draft in drafts
    ]
    return created + live


def keep_new_drafts(
    drafts: list[SuggestionDraft],
    *,
    taken: list[TakenItem],
    expert_ids: set[str],
    room: int,
) -> list[SuggestedTaskInput]:
    """The drafts worth storing: complete, not a restatement of a taken item
    or of each other, with an owner only when it is one of the user's
    experts."""
    seen = list(taken)
    kept: list[SuggestedTaskInput] = []
    for draft in drafts:
        if len(kept) >= room:
            break
        if not (draft.title.strip() and draft.brief.strip() and draft.reason.strip()):
            continue
        if any(_restates(draft, item) for item in seen):
            continue
        seen.append(TakenItem(title=draft.title, goal=draft.goal))
        owner = draft.owner_expert_id
        kept.append(
            SuggestedTaskInput(
                title=draft.title,
                goal=draft.goal,
                done_when=draft.done_when,
                brief=draft.brief,
                reason=draft.reason,
                owner_expert_id=owner if owner in expert_ids else None,
            )
        )
    return kept


def _restates(draft: SuggestionDraft, item: TakenItem) -> bool:
    return same_title(draft.title, item.title) or is_repeat(
        draft.title, draft.goal, item
    )


def same_title(a: str, b: str) -> bool:
    """Equal once case and punctuation are dropped, or nearly the same words."""
    words_a = set(_WORD_RE.findall(a.lower()))
    words_b = set(_WORD_RE.findall(b.lower()))
    if not words_a or not words_b:
        return a.strip().lower() == b.strip().lower()
    overlap = len(words_a & words_b) / len(words_a | words_b)
    return overlap >= _SAME_TITLE_OVERLAP


@attributed_usage
async def _ask_model(user_id: str, room: int, facts: str) -> SuggestionsResponse | None:
    """One bounded aux-tier call with a single retry; None when it failed."""
    messages = [
        {"role": "system", "content": system_prompt(room)},
        {"role": "user", "content": facts},
    ]
    correlation_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _TOTAL_BUDGET_SECONDS
    for attempt in range(_ATTEMPTS):
        remaining = deadline - loop.time()
        if remaining < _MIN_ATTEMPT_SECONDS:
            break
        job = InferenceJob(
            kind="briefing_narrative",
            phase="suggestions",
            correlation_id=correlation_id,
            latency_class="bounded",
            tier="aux",
            timeout_seconds=min(_TIMEOUT_SECONDS, remaining),
        )
        try:
            completion = await _complete(user_id, job, messages)
        except Exception as e:
            logger.warning(
                "Task suggestions attempt %s/%s failed: %s", attempt + 1, _ATTEMPTS, e
            )
            continue
        return completion.value
    return None


async def _complete(
    user_id: str, job: InferenceJob, messages: list[dict[str, str]]
) -> StructuredCompletion[SuggestionsResponse]:
    """The call on Otto's own scope; its spend is recorded whether or not the
    answer parsed."""
    scope = InferenceScope(user_id=user_id)
    ctx = InferenceContext(
        scope=scope, job=job, route=resolve_route(scope, job, config=config)
    )
    try:
        completion = await structured_complete(
            ctx, messages, SuggestionsResponse, max_output_tokens=_MAX_OUTPUT_TOKENS
        )
    except InferenceError as e:
        await _record_spend(ctx, e.usage)
        raise
    await _record_spend(ctx, completion.usage)
    return completion


async def _record_spend(ctx: InferenceContext, usage: InferenceUsage | None) -> None:
    """Charge the attempt as background work; never raises."""
    if usage is None:
        return
    try:
        await record(ctx, usage, block_name="copilot:briefing:suggestions")
    except Exception as e:
        logger.warning(
            "Task suggestions cost log failed for %s: %s", ctx.scope.user_id[:8], e
        )
