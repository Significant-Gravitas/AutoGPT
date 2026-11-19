"""Draft a simulated account's next tasks with production's own code.

``draft_task_suggestions`` runs as it is, with its database calls answered by
an in-memory account and its one model call sent to the eval's backend (or
answered by the case's scripted reply). The prompt, the facts block, the
room left under the cap and the code's own repeat guard are all
production's. The module's names are swapped for the length of one case, so
suggestion cases run one at a time.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

from prisma.enums import TaskCreator, TaskStatus
from pydantic import BaseModel, Field

from backend.api.features.experts.models import Expert
from backend.copilot.briefing import suggestions
from backend.copilot.briefing.suggestion_prompt import (
    SuggestionDraft,
    SuggestionsResponse,
    system_prompt,
)
from backend.copilot.eval.tasks.llm import Backend, first_json_object
from backend.data.task import Task
from backend.data.task_suggestion import SuggestedTaskInput

from .models import AccountSpec, BoardTask

NOW = datetime(2026, 11, 9, 8, 0, tzinfo=UTC)
USER_ID = "eval-suggestions-user"
_ONE_AT_A_TIME = asyncio.Lock()


class ModelCall(BaseModel):
    room: int
    facts: str
    raw: str = ""


class Drafting(BaseModel):
    """What one drafting run did."""

    returned: list[Task] = Field(default_factory=list)
    created: list[Task] = Field(default_factory=list)
    calls: list[ModelCall] = Field(default_factory=list)
    board: list[Task] = Field(default_factory=list)
    live_before: list[Task] = Field(default_factory=list)
    dismissed: list[str] = Field(
        default_factory=list, description="Dismissed and recently retired titles."
    )
    experts: dict[str, str] = Field(default_factory=dict)


class Drafter:
    """Where the drafting call goes: a backend and model, or a fixed reply."""

    def __init__(
        self,
        backend: Backend | None = None,
        model: str = "",
        scripted: list[dict[str, Any]] | None = None,
    ) -> None:
        self.backend, self.model, self.scripted = backend, model, scripted


async def draft_for(
    account: AccountSpec, world_experts: list[dict[str, Any]], drafter: Drafter
) -> Drafting:
    """Run production's drafting on the account and report what it did."""
    store = AccountStore(account, world_experts)
    calls: list[ModelCall] = []

    async def ask_model(
        user_id: str, room: int, facts: str
    ) -> SuggestionsResponse | None:
        call = ModelCall(room=room, facts=facts)
        calls.append(call)
        return await _draft(drafter, room, facts, call)

    async with _ONE_AT_A_TIME:
        with (
            patch.object(suggestions, "task_db", lambda: store),
            patch.object(suggestions, "task_suggestion_db", lambda: store),
            patch.object(suggestions, "experts_db", lambda: store),
            patch.object(suggestions, "_ask_model", ask_model),
        ):
            returned = await suggestions.draft_task_suggestions(USER_ID, NOW)
    return Drafting(
        returned=returned,
        created=store.created,
        calls=calls,
        board=store.board,
        live_before=store.live_before,
        dismissed=store.set_aside(),
        experts={e.id: e.name for e in store.experts},
    )


async def _draft(
    drafter: Drafter, room: int, facts: str, call: ModelCall
) -> SuggestionsResponse | None:
    if drafter.scripted is not None:
        drafts = [SuggestionDraft.model_validate(d) for d in drafter.scripted]
        return SuggestionsResponse(suggestions=drafts)
    backend = drafter.backend
    if backend is None:
        raise RuntimeError("This case needs a drafting model; run without --offline.")
    call.raw = await backend.text(drafter.model, system_prompt(room), facts, True)
    data = first_json_object(call.raw)
    return SuggestionsResponse.model_validate(data) if data is not None else None


class AccountStore:
    """The account's tasks, suggestions and experts, answering the calls
    drafting makes as the real data layer would."""

    def __init__(self, account: AccountSpec, world_experts: list[dict[str, Any]]):
        experts = world_experts if account.use_world_experts else []
        self.experts = [expert_of(e) for e in experts]
        self.board = [
            task_of(t, i, suggested=False) for i, t in enumerate(account.board)
        ]
        self.live = [
            task_of(t, i + 100, suggested=True) for i, t in enumerate(account.live)
        ]
        self.live_before = list(self.live)
        self.dismissed = list(account.dismissed)
        self.retired = list(account.retired)
        self.created: list[Task] = []

    async def list_tasks(
        self, user_id: str, limit: int = 50, suggested: bool = False, **_: Any
    ) -> list[Task]:
        rows = self.live if suggested else self.board
        return sorted(rows, key=lambda t: t.updated_at, reverse=True)[:limit]

    async def retire_stale_suggestions(
        self, user_id: str, created_before: datetime
    ) -> int:
        kept = [t for t in self.live if t.created_at >= created_before]
        self.retired += [t.title for t in self.live if t not in kept]
        retired = len(self.live) - len(kept)
        self.live = kept
        return retired

    async def dismissed_suggestion_titles(self, user_id: str) -> list[str]:
        """Dismissed ones and ones retired unstarted lately, as production
        lists them, so neither is drafted again."""
        return self.set_aside()

    def set_aside(self) -> list[str]:
        return [*self.dismissed, *self.retired]

    async def list_experts(self, user_id: str) -> list[Expert]:
        return list(self.experts)

    async def create_suggested_task(
        self, user_id: str, draft: SuggestedTaskInput
    ) -> Task:
        owner = draft.owner_expert_id
        if owner not in {e.id for e in self.experts}:
            owner = None
        task = Task(
            id=f"suggested_{len(self.created) + 1}",
            user_id=user_id,
            title=" ".join(draft.title.split()),
            description=draft.brief.strip() or None,
            goal=" ".join(draft.goal.split()) or None,
            done_when=" ".join(draft.done_when.split()) or None,
            status=TaskStatus.TODO,
            owner_expert_id=owner,
            created_by=TaskCreator.OTTO,
            created_at=NOW,
            updated_at=NOW,
            is_suggested=True,
            suggested_reason=" ".join(draft.reason.split()) or None,
        )
        self.created.append(task)
        self.live.append(task)
        return task


def task_of(spec: BoardTask, n: int, suggested: bool) -> Task:
    when = NOW - timedelta(days=spec.days_ago)
    status = TaskStatus(spec.status)
    settled = status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED)
    return Task(
        id=f"task_{n}",
        user_id=USER_ID,
        title=spec.title,
        status=status,
        owner_expert_id=spec.owner,
        created_by=TaskCreator.OTTO if suggested else TaskCreator.USER,
        created_at=when,
        updated_at=when,
        completed_at=when if settled else None,
        goal=spec.goal,
        result=spec.result,
        failure_reason=spec.failure_reason,
        status_text=spec.status_text,
        is_suggested=suggested,
    )


def expert_of(world_expert: dict[str, Any]) -> Expert:
    """A world's expert as the roster lists it."""
    return Expert(
        id=str(world_expert["id"]),
        name=str(world_expert.get("name", "")),
        avatar_url=None,
        role=str(world_expert.get("role", "")),
        job_title=str(world_expert.get("role", "")) or None,
        tagline=None,
        bio=str(world_expert.get("owns", "")) or None,
        skills=[],
        identity="",
        voice_preferences="",
        boundaries="",
        protected_soul_rules=[],
        is_template=False,
        source_template_id=None,
        is_archived=False,
        workflows=[],
    )
