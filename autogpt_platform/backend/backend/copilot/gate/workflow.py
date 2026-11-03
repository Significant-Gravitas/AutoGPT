"""The gate answers the review pause of a workflow step AutoPilot started.

The step parks as a human review always does; the gate then judges it as it
would a direct call of the same block in the chat that started the run. A run
verdict approves the row at once; an ask opens the gate's card in that chat,
pointing at the row, and answering either closes the other. Every failure
leaves an ordinary human review.
"""

import logging
from collections.abc import Iterable
from typing import Any

from prisma.enums import ReviewStatus
from pydantic import BaseModel

from backend.api.features.graph_executions.review.model import PendingHumanReviewModel
from backend.blocks._base import Block
from backend.copilot.constants import (
    AUTOPILOT_NAME,
    COPILOT_STEP_CARD_PREFIX,
    step_card_id,
)
from backend.copilot.model import (
    ChatSession,
    get_chat_session,
    get_chat_session_metadata,
)
from backend.copilot.rate_limit import is_user_paywalled
from backend.data.db_accessors import graph_db, review_db
from backend.data.execution import ExecutionContext

from . import Ruling, _judged_args, _typed_messages, chat_rules, gate_active
from . import review as review_store
from . import rule_on, unified_approvals
from .classifier import Judgement, supervise
from .policy import DEFAULT_MODE, Effect
from .subject import Subject, block_subject

logger = logging.getLogger(__name__)

STEP_TOOL = "workflow_step"
# Decision 2 (OPEN-3592): a workflow's own "ask before irreversible actions"
# is a floor, so the mode cannot run its steps; a rule on the block still can.
WORKFLOW_SETTING_IS_A_FLOOR = True
APPROVED_IN_CHAT = "Approved in chat"
DECLINED_IN_CHAT = "Declined in chat"


async def review_route(ctx: ExecutionContext, user_id: str) -> ChatSession | None:
    """The chat whose gate answers this pause, or None for a human review."""
    if not ctx.approval_chat_id:
        return None
    try:
        if not await unified_approvals(user_id):
            return None
        # Without a plan the step must wait for a resume, which the paywall answers.
        if await is_user_paywalled(user_id):
            return None
        session = await get_chat_session(ctx.approval_chat_id, user_id)
        if session is None or not await gate_active(user_id, session):
            return None
        return session
    except Exception:
        logger.warning(
            f"Gate could not route the review of run {ctx.graph_exec_id}",
            exc_info=True,
        )
        return None


async def judge_step(
    session: ChatSession,
    block: Block,
    inputs: dict[str, Any],
    ctx: ExecutionContext,
    step: review_store.WorkflowStep,
) -> Ruling:
    """What the chat's gate says about this step, as about a direct call of its
    block: the user's rule on the block, else the mode, else the supervisor."""
    subject = step_subject(block, inputs, step.name)

    async def judge() -> Judgement:
        # Judged as a direct call of the block would be: same rubric, vouched name.
        outward_judged = subject.effect is Effect.EXTERNAL_JUDGED
        args = {"workflow": step.name, "step": subject.name, "input": inputs}
        return await supervise(
            tool_name=STEP_TOOL,
            args=_judged_args(args, None, subject.name if outward_judged else None),
            user_messages=request_that_started(
                session, ctx.root_execution_id or step.graph_exec_id
            ),
            outward_judged=outward_judged,
        )

    ruling = await rule_on(
        session,
        session.user_id,
        subject.key,
        subject.effect,
        subject,
        judge,
        # A context from before the setting was carried counts it as on.
        floor=WORKFLOW_SETTING_IS_A_FLOOR
        and ctx.graph_sensitive_action_safe_mode is not False,
    )
    if ruling.run:
        return ruling.model_copy(update={"reason": _allowed_by(ruling, subject)})
    return ruling


async def open_step_card(
    session: ChatSession,
    block: Block,
    inputs: dict[str, Any],
    step: review_store.WorkflowStep,
    ruling: Ruling,
) -> bool:
    """The gate's card in the chat that started the run, pointing at the
    step's row; False when it could not be opened."""
    user_id = session.user_id
    try:
        payload = step_card_payload(
            block,
            inputs,
            step,
            ruling,
            mode=session.metadata.autopilot_mode or DEFAULT_MODE,
            turn=review_store.turn_of(session),
        )
    except Exception:
        logger.warning(
            f"Gate could not build the card for step {step.node_exec_id}",
            exc_info=True,
        )
        return False
    return await review_store.open_review_row(
        step_card_id(user_id, step.node_exec_id),
        user_id,
        session,
        payload,
        review_store.payload_headline(payload),
    )


def step_card_payload(
    block: Block,
    inputs: dict[str, Any],
    step: review_store.WorkflowStep,
    ruling: Ruling,
    *,
    mode: str,
    turn: int,
) -> dict[str, Any]:
    """The card a direct call of the block would show, naming its workflow."""
    return review_store.review_payload(
        STEP_TOOL,
        {"input": inputs},
        step_subject(block, inputs, step.name),
        reason=ruling.reason,
        reason_kind=ruling.reason_kind,
        decided_by=ruling.decided_by,
        mode=mode,
        turn=turn,
        workflow=step,
    )


class StepAnswers(BaseModel):
    # The steps whose rows took their card's answer, for the route to resume.
    answered: list[review_store.WorkflowStep] = []
    # Cards whose step could not take the answer; each stays for a retry.
    failed: list[str] = []


async def finish_step_cards(
    user_id: str, cards: Iterable[PendingHumanReviewModel]
) -> StepAnswers:
    """Answer each answered workflow card's row the way the card was answered.
    A card is consumed once its row has the answer, or already had another."""
    finished = StepAnswers()
    for card in cards:
        step = step_of(card)
        if step is None or card.status == ReviewStatus.WAITING:
            continue
        approved = card.status == ReviewStatus.APPROVED
        try:
            # No reviewed data: the step runs on the input its card showed.
            await review_db().process_all_reviews_for_execution(
                user_id=user_id,
                review_decisions={
                    step.node_exec_id: (
                        card.status,
                        None,
                        APPROVED_IN_CHAT if approved else DECLINED_IN_CHAT,
                    )
                },
            )
            finished.answered.append(step)
        except ValueError:
            # Answered differently on the run page first: that answer stands.
            logger.warning(
                f"Step {step.node_exec_id} kept its answer from the run page",
                exc_info=True,
            )
        except Exception:
            logger.warning(
                f"Step {step.node_exec_id} could not take its card's answer",
                exc_info=True,
            )
            finished.failed.append(card.node_exec_id)
            await _reopen(user_id, card.node_exec_id)
            continue
        await review_store.consume(card.node_exec_id, user_id)
        if not approved and card.session_id:
            await _ask_from_now_on(user_id, card)
    return finished


async def close_step_cards(user_id: str, node_exec_ids: Iterable[str]) -> None:
    """The steps were answered on the run page, so their cards leave the chat."""
    for node_exec_id in node_exec_ids:
        await review_store.consume(step_card_id(user_id, node_exec_id), user_id)


async def card_chat_for_run(user_id: str, graph_exec_id: str) -> str | None:
    """The chat holding a card for one of this run's waiting steps, if any."""
    try:
        rows = await review_db().get_pending_reviews_for_execution(
            graph_exec_id, user_id
        )
        cards = await review_db().get_reviews_by_node_exec_ids(
            [step_card_id(user_id, row.node_exec_id) for row in rows], user_id
        )
    except Exception:
        logger.warning(f"Could not read step cards of run {graph_exec_id}")
        return None
    return next(
        (
            card.session_id
            for card in cards.values()
            if card.status == ReviewStatus.WAITING and card.session_id
        ),
        None,
    )


def awaiting_whom(card_chat: str | None, session_id: str | None) -> str:
    """How a parked run's state reads to the model that asks about it."""
    if card_chat is None:
        return "is awaiting human review"
    if card_chat == session_id:
        return "is waiting for the user's approval in this chat"
    return "is waiting for the user's approval in the chat that started it"


async def workflow_step(
    graph_exec_id: str | None,
    graph_id: str | None,
    graph_version: int | None,
    node_exec_id: str,
) -> review_store.WorkflowStep | None:
    """What the card says about the step, from the pause's own ids: a
    sub-graph's context still names its parent's run. None outside a run."""
    if not (graph_exec_id and graph_id and graph_version is not None):
        return None
    try:
        graph = await graph_db().get_graph_metadata(graph_id, graph_version)
        name = graph.name if graph and graph.name else ""
    except Exception:
        logger.warning(f"Could not name workflow {graph_id}", exc_info=True)
        name = ""
    return review_store.WorkflowStep(
        graph_exec_id=graph_exec_id,
        node_exec_id=node_exec_id,
        graph_id=graph_id,
        graph_version=graph_version,
        name=name or "Untitled workflow",
    )


def step_subject(block: Block, inputs: dict[str, Any], workflow: str) -> Subject:
    """The block's own subject, so a rule set from either kind of card covers
    both, with the card's reason naming the workflow."""
    subject = block_subject(block, inputs, priced=False)
    return subject.model_copy(
        update={"reason": f"A step of {workflow}, which reaches outside the platform."}
    )


def step_of(card: PendingHumanReviewModel) -> review_store.WorkflowStep | None:
    if not is_step_card(card):
        return None
    payload = card.payload if isinstance(card.payload, dict) else {}
    try:
        return review_store.WorkflowStep.model_validate(payload.get("workflow"))
    except Exception:
        logger.warning(f"Workflow card {card.node_exec_id} names no step")
        return None


def is_step_card(card: PendingHumanReviewModel) -> bool:
    return card.node_exec_id.startswith(COPILOT_STEP_CARD_PREFIX)


def step_rule_keys(cards: Iterable[PendingHumanReviewModel]) -> dict[str, str]:
    """The subject each workflow card can set a rule on, from the card the
    server wrote: a step has no held call to read it from."""
    keys: dict[str, str] = {}
    for card in cards:
        payload = card.payload if isinstance(card.payload, dict) else {}
        subject = payload.get("subject")
        if is_step_card(card) and isinstance(subject, dict) and subject.get("key"):
            keys[card.node_exec_id] = str(subject["key"])
    return keys


def request_that_started(session: ChatSession, run_id: str | None) -> list[str]:
    """What the user typed up to the call that started the run: a run parked
    for days must not be judged against whatever they said since."""
    started = next(
        (
            index
            for index, message in enumerate(session.messages)
            if run_id
            and message.role == "tool"
            and message.content
            and run_id in message.content
        ),
        None,
    )
    if started is None:
        return []
    return _typed_messages(session, upto=started)


async def _reopen(user_id: str, card_id: str) -> None:
    # Back in the chat's queue, so the user can answer it again.
    try:
        await review_db().reopen_review(card_id, user_id)
    except Exception:
        logger.warning(f"Could not reopen card {card_id}", exc_info=True)


async def _ask_from_now_on(user_id: str, card: PendingHumanReviewModel) -> None:
    # As a declined direct call does: the block asks for the rest of the chat.
    key = step_rule_keys([card]).get(card.node_exec_id)
    if not key or not card.session_id:
        return
    try:
        chat = await get_chat_session_metadata(card.session_id, user_id)
        await chat_rules.set_ask(
            card.session_id, key, user_id, chat.expert_id if chat else None
        )
    except Exception:
        # The step already has its answer; a lost rule must not strand its run.
        logger.warning(
            f"Could not set an ask rule from card {card.node_exec_id}",
            exc_info=True,
        )


def _allowed_by(ruling: Ruling, subject: Subject) -> str:
    """The audit line on the step's approved row, read on the run page."""
    if ruling.by == "rule":
        why = f"you set {subject.name} to run without asking"
    elif ruling.by == "supervisor":
        why = "its check found this step matches your request"
    else:
        why = "Unsupervised is on in the chat that started this run"
    return f"Allowed by {AUTOPILOT_NAME}: {why}."
