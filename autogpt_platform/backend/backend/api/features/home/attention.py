import json
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from backend.api.features.executions.review.model import PendingHumanReviewModel
from backend.api.features.experts.models import Expert
from backend.executor.scheduler import CopilotTurnJobInfo, GraphExecutionJobInfo

from .helpers import as_utc, run_link, setup_count, to_home_expert
from .models import HomeAction, HomeAttentionItem, HomeExpert

# Longest payload preview we return before clipping it with an ellipsis.
_PREVIEW_MAX = 140


def compose_attention_items(
    *,
    now: datetime,
    experts: list[Expert],
    reviews: list[PendingHumanReviewModel],
    schedules: list[GraphExecutionJobInfo | CopilotTurnJobInfo],
    credits_balance: int | None,
) -> list[HomeAttentionItem]:
    items = [_review_attention(review, now) for review in reviews]
    items.extend(
        _expert_attention(expert) for expert in experts if _needs_attention(expert)
    )
    if credits_balance is not None and credits_balance <= 0 and schedules:
        items.append(_credits_attention(len(schedules)))
    return sorted(items, key=_attention_sort_key)


def _review_attention(
    review: PendingHumanReviewModel, now: datetime
) -> HomeAttentionItem:
    title = review.instructions or review.agent_name or "Review an agent decision"
    created_at = as_utc(review.created_at)
    return HomeAttentionItem(
        id=f"approval-{review.node_exec_id}",
        kind="approval",
        priority=("high" if now - created_at > timedelta(hours=24) else "normal"),
        title=title,
        description="Your agent paused before taking an external action.",
        why_it_matters="The task cannot continue until you approve or decline it.",
        expert=_review_expert(review),
        agent_name=review.agent_name,
        created_at=created_at,
        preview=_payload_preview(review.payload),
        review=review,
        primary_action=HomeAction(label="Review", href=_review_link(review)),
    )


def _expert_attention(expert: Expert) -> HomeAttentionItem:
    summary = to_home_expert(expert)
    if expert.schedules_paused_at:
        budget = expert.weekly_budget
        description = (
            f"Weekly budget reached: {expert.weekly_spend} of {budget} credits."
            if budget is not None and expert.weekly_spend >= budget
            else "Scheduled work is paused."
        )
        return HomeAttentionItem(
            id=f"paused-{expert.id}",
            kind="paused",
            priority="high",
            title=f"Review {expert.name}'s paused work",
            description=description,
            why_it_matters="Upcoming tasks will not run while this agent is paused.",
            expert=summary,
            created_at=as_utc(expert.schedules_paused_at),
            primary_action=HomeAction(
                label="Review budget", href=f"/team/{quote(expert.id)}"
            ),
        )
    count = setup_count(expert)
    return HomeAttentionItem(
        id=f"setup-{expert.id}",
        kind="setup",
        priority="normal",
        title=f"Finish setting up {expert.name}",
        description=(
            "1 scheduled workflow needs setup."
            if count == 1
            else f"{count} scheduled workflows need setup."
        ),
        why_it_matters="Those workflows cannot run until their connections are ready.",
        expert=summary,
        primary_action=HomeAction(
            label="Finish setup", href=f"/team/{quote(expert.id)}"
        ),
    )


def _credits_attention(schedule_count: int) -> HomeAttentionItem:
    return HomeAttentionItem(
        id="credits",
        kind="credits",
        priority="high",
        title="Add credits for scheduled work",
        description=(
            f"{schedule_count} upcoming task{'s' if schedule_count != 1 else ''} may not run."
        ),
        why_it_matters="Agents need a positive balance before paid blocks can start.",
        primary_action=HomeAction(label="Add credits", href="/profile/credits"),
    )


def _review_expert(review: PendingHumanReviewModel) -> HomeExpert | None:
    if not review.expert_id or not review.expert_name:
        return None
    return HomeExpert(
        id=review.expert_id,
        name=review.expert_name,
        role="Agent",
        avatar_url=review.expert_avatar_url,
    )


def _needs_attention(expert: Expert) -> bool:
    return bool(expert.schedules_paused_at or setup_count(expert) > 0)


def _payload_preview(payload: object) -> str | None:
    if payload is None:
        return None
    rendered = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    compact = " ".join(rendered.split())
    if len(compact) <= _PREVIEW_MAX:
        return compact
    return f"{compact[:_PREVIEW_MAX - 3]}…"


def _attention_sort_key(item: HomeAttentionItem) -> tuple[int, datetime]:
    priority = 0 if item.priority == "high" else 1
    created = item.created_at or datetime.max.replace(tzinfo=timezone.utc)
    return priority, created


def _review_link(review: PendingHumanReviewModel) -> str:
    if review.session_id:
        return f"/copilot?sessionId={quote(review.session_id)}"
    if review.library_agent_id:
        return run_link(review.library_agent_id, review.graph_exec_id) or "/library"
    return "/library"
