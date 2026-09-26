"""Ask a teammate to audit a piece of work before it goes out.

``delegate_to_expert`` borrows a teammate to *do* something: it opens a
``ChatSession`` and runs a full agentic turn under their identity, memory and
budget. That is the right shape for work and the wrong shape for a check — a
check that costs a session is a check the model skips.

A consult is one bounded LLM call. No session, no tools, no memory, no thread.
Recursion is impossible because there is no tool surface to recurse through,
and the expert memory boundary does not move: what crosses between two experts
is exactly the two strings the caller wrote. The audit frame itself lives in
``consult_audit``, which explains why it is not a persona.

The caller must state the ``authority`` behind its own draft. Asking a reviewer
that cannot see the caller's conversation or memory to *find* the approval
instead would block every legitimate commitment, and would make "paste your
whole context in" the only way to pass — which is the boundary erosion this
tool is careful not to cause.

Every failure path — provider down, unparseable JSON, timeout, dry run —
returns ``insufficient``, never an error the caller can shrug off and never
``pass``. A check that did not happen must not read as one that did.
"""

import logging
from typing import Any

from backend.api.features.experts.models import Expert
from backend.copilot.config import ChatConfig
from backend.copilot.context import take_consult_slot
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
from backend.copilot.inference.trace import trace
from backend.copilot.model import ChatSession

from .base import BaseTool
from .consult_audit import (
    MAX_OUTPUT_TOKENS,
    TIMEOUT_SECONDS,
    VerdictPayload,
    audit_frame,
    audit_material,
    verdict_response,
)
from .expert_delegation import resolve_target_expert, unknown_target_message
from .models import ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)

config = ChatConfig()


class ConsultTeammateTool(BaseTool):
    """Get a bounded, structured second opinion from another expert."""

    @property
    def name(self) -> str:
        return "consult_teammate"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Ask a teammate to CHECK a piece of work: pass / block / "
            "insufficient, with the offending lines quoted. Use before any "
            "commitment (money, dates, refunds, guarantees) leaves the "
            "conversation — state the authority it rests on. Use "
            "delegate_to_expert when a teammate should DO the work instead."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expert_id": {
                    "type": "string",
                    "description": "Teammate to ask: expert id from <team_context>, or name. Not you.",
                },
                "work": {
                    "type": "string",
                    "description": "The work to check, verbatim.",
                },
                "authority": {
                    "type": "string",
                    "description": (
                        "What each commitment rests on — user approval, "
                        "done work, a system confirmation. 'none' if it "
                        "commits to nothing."
                    ),
                },
                "question": {
                    "type": "string",
                    "description": "Extra thing to rule on ('yes' = a problem). The commitment check always runs.",
                    "default": "",
                },
            },
            "required": ["expert_id", "work", "authority"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        expert_id: str = "",
        work: str = "",
        authority: str = "",
        question: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        if user_id is None:
            return self._error("Authentication required", session)
        target_id = expert_id.strip()
        if not target_id:
            return self._error("expert_id is required", session)
        if not work.strip():
            return self._error("work is required — paste what would be sent", session)
        if not authority.strip():
            return self._error(
                "authority is required — list what each commitment in the "
                "work rests on, or 'none' if it commits to nothing.",
                session,
            )
        if target_id == session.expert_id:
            return self._error(_SELF_REFUSAL, session)

        reviewer = await self._resolve_reviewer(user_id, target_id, session)
        if isinstance(reviewer, ErrorResponse):
            return reviewer

        refusal = take_consult_slot()
        if refusal is not None:
            return self._error(refusal, session)

        verdict = await _audit_via_provider(
            user_id, session, reviewer, work, authority, question
        )
        return verdict_response(reviewer, verdict, session)

    def _error(self, message: str, session: ChatSession) -> ErrorResponse:
        return ErrorResponse(message=message, session_id=session.session_id)

    async def _resolve_reviewer(
        self, user_id: str, target_id: str, session: ChatSession
    ) -> Expert | ErrorResponse:
        """Resolve the teammate. A paused expert may still give an opinion.

        ``delegate_to_expert`` refuses a paused teammate because delegation
        makes them spend and act; an audit starts nothing on their side, and
        withholding a check from a team already under budget pressure is
        backwards.
        """
        try:
            reviewer = await resolve_target_expert(user_id, target_id)
        except Exception as e:
            logger.warning(f"Consult target lookup failed for {target_id}: {e}")
            return self._error(
                "Could not reach that teammate right now. Try again.", session
            )
        if reviewer is None or reviewer.is_archived:
            return self._error(
                await unknown_target_message(user_id, target_id, session.expert_id),
                session,
            )
        if reviewer.id == session.expert_id:
            # A name reference can resolve back to the caller past the id check.
            return self._error(_SELF_REFUSAL, session)
        return reviewer


async def _audit_via_provider(
    user_id: str,
    session: ChatSession,
    reviewer: Expert,
    work: str,
    authority: str,
    question: str,
) -> VerdictPayload:
    """One completion against the fixed audit frame. Never raises.

    A dry-run session simulates side effects rather than paying for them, so it
    must not buy a verdict — nor report one it did not get.
    """
    if session.dry_run:
        return _not_checked(
            f"This is a dry-run session, so {reviewer.name} was not asked."
        )
    messages = [
        {"role": "system", "content": audit_frame(reviewer)},
        {"role": "user", "content": audit_material(work, authority, question)},
    ]
    try:
        completion = await _complete(user_id, session, messages)
    except InferenceError as e:
        logger.warning(f"Consult of {reviewer.id} did not parse: {e}")
        return _not_checked(f"{reviewer.name} could not be reached for a verdict.")
    except Exception as e:
        logger.warning(f"Consult of {reviewer.id} failed: {e}")
        return _not_checked(f"{reviewer.name} could not be reached for a verdict.")
    return completion.value


async def _complete(
    user_id: str, session: ChatSession, messages: list[dict[str, str]]
) -> StructuredCompletion[VerdictPayload]:
    """The audit call, traced as part of the asking chat's turn.

    On the cheap aux model, not the turn's own. The audit is extraction
    ("what does this promise that the authority list does not cover"), and the
    expensive model here is the one that wrote the draft — its judgement is
    what is under test, so re-asking it buys nothing. The spend belongs to
    the asking chat and its expert, and is recorded whether or not the answer
    parsed.
    """
    scope = InferenceScope(user_id=user_id, expert_id=session.expert_id)
    job = InferenceJob(
        kind="consult",
        correlation_id=session.session_id,
        latency_class="bounded",
        tier="aux",
        timeout_seconds=TIMEOUT_SECONDS,
    )
    ctx = InferenceContext(
        scope=scope, job=job, route=resolve_route(scope, job, config=config)
    )
    async with trace(ctx) as call:
        try:
            completion = await structured_complete(
                call.ctx, messages, VerdictPayload, max_output_tokens=MAX_OUTPUT_TOKENS
            )
        except InferenceError as e:
            # A response that arrived but didn't parse was still billed.
            await _record_spend(call.ctx, session, e.usage)
            raise
        call.usage = await _record_spend(call.ctx, session, completion.usage)
    return completion


_SELF_REFUSAL = (
    "You are that expert. A check has to come from someone else — reviewing "
    "your own draft in your own context is what this tool exists to replace."
)


def _not_checked(reason: str) -> VerdictPayload:
    return VerdictPayload(
        verdict="insufficient",
        reason=f"{reason} This work has NOT been checked; that is not approval.",
    )


async def _record_spend(
    ctx: InferenceContext, session: ChatSession, usage: InferenceUsage | None
) -> InferenceUsage | None:
    """Book the audit's spend against the user, like any other turn cost;
    returns it as priced.

    Never raises: a cost-ledger write failing must not turn a verdict the user
    already paid for into a tool error.
    """
    if usage is None:
        return None
    try:
        return await record(
            ctx, usage, block_name="copilot:consult_teammate", session=session
        )
    except Exception as e:
        logger.warning(f"Consult cost log failed for {ctx.scope.user_id[:8]}: {e}")
        return usage
