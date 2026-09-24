import json
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from pydantic import ValidationError

from backend.api.features.experts.models import Expert
from backend.api.features.experts.spend_approval import is_spend_review
from backend.api.features.graph_executions.review.model import PendingHumanReviewModel
from backend.copilot.briefing.outcome import as_utc, run_link
from backend.copilot.constants import AUTOPILOT_NAME
from backend.copilot.gate.review import GATE_NODE_PREFIX, GateReviewPayload
from backend.copilot.model import ChatSessionInfo, PendingQuestion
from backend.executor.scheduler import CopilotTurnJobInfo, GraphExecutionJobInfo

from .helpers import setup_count, to_home_expert
from .models import HomeAction, HomeAttentionItem, HomeExpert, HomeHeadline

# Longest payload preview we return before clipping it with an ellipsis.
_PREVIEW_MAX = 140


def compose_attention_items(
    *,
    now: datetime,
    experts: list[Expert],
    reviews: list[PendingHumanReviewModel],
    schedules: list[GraphExecutionJobInfo | CopilotTurnJobInfo],
    credits_balance: int | None,
    questions: list[ChatSessionInfo] | None = None,
) -> list[HomeAttentionItem]:
    expert_by_id = {expert.id: expert for expert in experts}
    items = [_review_attention(review, now) for review in reviews]
    items.extend(
        _expert_attention(expert) for expert in experts if _needs_attention(expert)
    )
    items.extend(
        _question_attention(session, question, expert_by_id)
        for session, question in _open_questions(questions or [])
        if _is_answerable(session, expert_by_id)
    )
    if credits_balance is not None and credits_balance <= 0 and schedules:
        items.append(_credits_attention(len(schedules)))
    return sorted(items, key=_attention_sort_key)


def _review_attention(
    review: PendingHumanReviewModel, now: datetime
) -> HomeAttentionItem:
    if gate := _gate_payload(review):
        return _gate_attention(review, gate, now)
    title = (
        review.action
        or review.instructions
        or review.agent_name
        or "Review a workflow step"
    )
    created_at = as_utc(review.created_at)
    if is_spend_review(review.node_exec_id):
        description = "Spending threshold reached; this work is on hold."
        why_it_matters = "It runs once you approve; declining cancels it."
    else:
        description = f"{_waiting_on(review)} is waiting for your approval."
        why_it_matters = "The task cannot continue until you approve or decline it."
    return HomeAttentionItem(
        id=f"approval-{review.node_exec_id}",
        kind="approval",
        priority=("high" if now - created_at > timedelta(hours=24) else "normal"),
        title=title,
        description=description,
        why_it_matters=why_it_matters,
        expert=_review_expert(review),
        agent_name=review.agent_name,
        created_at=created_at,
        preview=_payload_preview(review.payload),
        review=review,
        primary_action=HomeAction(label="Review", href=_review_link(review)),
    )


def _gate_attention(
    review: PendingHumanReviewModel, gate: GateReviewPayload, now: datetime
) -> HomeAttentionItem:
    """A held AutoPilot call: the card's own headline and reason, and its inputs
    as the preview, because Home answers it without opening the chat."""
    created_at = as_utc(review.created_at)
    return HomeAttentionItem(
        id=f"approval-{review.node_exec_id}",
        kind="approval",
        priority=("high" if now - created_at > timedelta(hours=24) else "normal"),
        title=gate.headline.text,
        headline=HomeHeadline(ask=gate.headline.ask, object=gate.headline.object),
        description=_gate_reason(gate),
        why_it_matters="Nothing runs until you approve it.",
        expert=_review_expert(review),
        created_at=created_at,
        preview=_clip(
            " · ".join(
                f"{field.label}: {_named_value(gate, field.key) or _preview_value(gate.arguments[field.key])}"
                for field in gate.fields
                if field.key != gate.headline.object_key
                and gate.arguments.get(field.key) not in (None, "", [], {})
            )
        )
        or None,
        review=review,
        primary_action=HomeAction(label="Open chat", href=_review_link(review)),
    )


def _gate_payload(review: PendingHumanReviewModel) -> GateReviewPayload | None:
    if not review.node_exec_id.startswith(GATE_NODE_PREFIX):
        return None
    try:
        return GateReviewPayload.model_validate(review.payload)
    except ValidationError:
        # A row written before the payload carried a headline.
        return None


def _gate_reason(gate: GateReviewPayload) -> str:
    if gate.reason_kind == "supervisor" and gate.reason:
        return f"Not sure this is safe: {gate.reason}"
    if gate.reason_kind in ("subject", "rule", "content") and gate.reason:
        return gate.reason
    return f"{AUTOPILOT_NAME} is waiting for your approval."


def _named_value(gate: GateReviewPayload, key: str) -> str | None:
    """An id argument as the names it resolved to, as the card shows it."""
    refs = [ref for ref in gate.references if ref.key == key]
    if not any(ref.name for ref in refs):
        return None
    value = gate.arguments.get(key)
    more = len(value) - len(refs) if isinstance(value, list) else 0
    text = ", ".join(ref.name or ref.id for ref in refs)
    return f"{text} +{more} more" if more > 0 else text


def _preview_value(value: object) -> str:
    """As the card shows it: a list of names joined, never a Python repr."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(
        isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in value
    ):
        return ", ".join(str(v) for v in value)
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return json.dumps(value, default=str, ensure_ascii=False)


def _waiting_on(review: PendingHumanReviewModel) -> str:
    if review.session_id:
        return AUTOPILOT_NAME
    if review.agent_name:
        return f"Workflow “{review.agent_name}”"
    return "A workflow"


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
        # The Team page carries the setup card that names and fixes the gap.
        primary_action=HomeAction(label="Finish setup", href="/team"),
    )


def _open_questions(
    sessions: list[ChatSessionInfo],
) -> list[tuple[ChatSessionInfo, PendingQuestion]]:
    """The questions still waiting on the user, one row per session.

    Answering clears ``pending_question``, so a session without one has
    nothing left to show. A session that asked twice keeps only its latest
    question, and collapses to a single row here even if the caller hands us
    the same session more than once — the item id is keyed on the session, so
    two rows would be two cards with the same id.
    """
    latest: dict[str, tuple[ChatSessionInfo, PendingQuestion]] = {}
    for session in sessions:
        question = session.metadata.pending_question
        if question is None:
            continue
        seen = latest.get(session.session_id)
        if seen is None or seen[1].asked_at <= question.asked_at:
            latest[session.session_id] = (session, question)
    return list(latest.values())


def _question_attention(
    session: ChatSessionInfo,
    question: PendingQuestion,
    expert_by_id: dict[str, Expert],
) -> HomeAttentionItem:
    """A chat that ended waiting on the user. Replying there is the only fix,
    so the item links straight back into the thread and has no own action."""
    asker = expert_by_id.get(session.expert_id) if session.expert_id else None
    return HomeAttentionItem(
        id=f"question-{session.session_id}",
        kind="question",
        priority="normal",
        title=f"{asker.name if asker else AUTOPILOT_NAME} has a question",
        description=_clip(question.text),
        why_it_matters="The work is paused until you answer in the chat.",
        expert=to_home_expert(asker) if asker else None,
        created_at=as_utc(question.asked_at),
        primary_action=HomeAction(
            label="Answer",
            href=f"/copilot?sessionId={quote(session.session_id)}",
        ),
    )


def _is_answerable(session: ChatSessionInfo, expert_by_id: dict[str, Expert]) -> bool:
    """Whether replying in the thread can still clear the question.

    ``expert_by_id`` holds only active experts, so a missing entry means the
    asking teammate was archived. That thread now fails its expert-identity
    build before the reply reaches ``clear_pending_question``, so the card
    would survive every answer the user gives it and nothing else on Home can
    dismiss it. Dropping it is the only way it ever goes away.
    """
    return session.expert_id is None or session.expert_id in expert_by_id


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
    return _clip(rendered)


def _clip(text: str) -> str:
    compact = " ".join(text.split())
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
    if review.library_agent_id and review.graph_exec_id:
        return run_link(review.library_agent_id, review.graph_exec_id) or "/library"
    return "/library"
