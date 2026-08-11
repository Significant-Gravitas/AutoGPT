"""The alert cause catalog.

An Alert fires only for states where the platform is blocked on the human. Each
cause carries exactly the slots its sentences need, and knows how to render
itself three ways: as the primary block of an Alert, as a secondary "also
waiting" line when several alerts coalesce into one email, and as an attention
card in the Briefing when it was deduped, capped or has persisted.

Every sentence is assembled from those slots — there is no hand-written copy
per condition — and the CTA verb is always the fix, never "View dashboard".
"""

from typing import Annotated, Literal, Union

from prisma.enums import AlertCause
from pydantic import BaseModel, Field

from backend.data.notifications import AlertAlsoItem, AlertFact, BriefingAttentionItem

# Blocked-on-everything first, forecasts last. The order is load-bearing: it
# picks which condition leads a coalesced Alert, and the Briefing's first
# attention card gets the strong amber rule.
SEVERITY: dict[AlertCause, int] = {
    AlertCause.ZERO_BALANCE: 0,
    AlertCause.AUTH_EXPIRED: 1,
    AlertCause.PAUSED_FAILURES: 2,
    AlertCause.CONTINUOUS_ERROR: 3,
    AlertCause.BLOCK_FAILED: 4,
    AlertCause.GUARDRAIL: 5,
    AlertCause.AWAITING_INPUT: 6,
    AlertCause.AWAITING_REVIEW: 7,
    AlertCause.LOW_BALANCE: 8,
}


class BaseCause(BaseModel):
    """Slots plus the four renderings every cause owes the email system."""

    cause: AlertCause
    # Relative path into the platform; the base URL is applied at render time
    # so a stored condition still links correctly after a domain change.
    cta_path: str

    @property
    def subject_line(self) -> str:
        return self.headline

    @property
    def headline(self) -> str:
        raise NotImplementedError

    @property
    def body(self) -> str:
        raise NotImplementedError

    @property
    def cta_label(self) -> str:
        raise NotImplementedError

    @property
    def agent_label(self) -> str:
        """Name shown in bold on a coalesced line and in the Briefing card."""
        raise NotImplementedError

    @property
    def tag(self) -> str | None:
        """Short all-caps chip on the Briefing attention card."""
        return None

    @property
    def facts(self) -> list[AlertFact]:
        """Used when the load-bearing detail is data rather than prose."""
        return []

    @property
    def microcopy(self) -> str | None:
        return None

    def also_item(self, base_url: str) -> AlertAlsoItem:
        return AlertAlsoItem(
            agent=self.agent_label,
            text=self.body,
            link_label=self.cta_label,
            url=f"{base_url}{self.cta_path}",
        )

    def attention_item(self, base_url: str) -> BriefingAttentionItem:
        return BriefingAttentionItem(
            agent=self.agent_label,
            title=self.headline,
            tag=self.tag,
            body=self.body,
            cta_label=self.cta_label,
            cta_url=f"{base_url}{self.cta_path}",
        )


class AuthExpiredCause(BaseCause):
    cause: Literal[AlertCause.AUTH_EXPIRED] = AlertCause.AUTH_EXPIRED
    agent: str
    provider: str
    expired_at_label: str
    runs_skipped: int
    next_try_label: str

    @property
    def headline(self) -> str:
        return f"{self.agent} is stuck"

    @property
    def subject_line(self) -> str:
        return f"{self.agent} is stuck — {self.provider} needs a reconnect"

    @property
    def body(self) -> str:
        runs = "1 scheduled run has" if self.runs_skipped == 1 else f"{self.runs_skipped} scheduled runs have"
        return (
            f"{self.provider}’s connection expired at {self.expired_at_label}, so "
            f"{self.agent} can’t run. {runs} been skipped; the next try is at "
            f"{self.next_try_label}. Until then, its schedule is on hold."
        )

    @property
    def cta_label(self) -> str:
        return f"Reconnect {self.provider}"

    @property
    def agent_label(self) -> str:
        return self.agent

    @property
    def tag(self) -> str | None:
        return f"{self.runs_skipped} runs skipped" if self.runs_skipped else None

    @property
    def microcopy(self) -> str | None:
        return "Usually takes under a minute"


class PausedFailuresCause(BaseCause):
    cause: Literal[AlertCause.PAUSED_FAILURES] = AlertCause.PAUSED_FAILURES
    agent: str
    step: str
    consecutive_failures: int

    @property
    def headline(self) -> str:
        return f"{self.agent} paused itself"

    @property
    def body(self) -> str:
        return (
            f"The step “{self.step}” failed on {self.consecutive_failures} "
            "consecutive runs. It stays paused until you take a look."
        )

    @property
    def cta_label(self) -> str:
        return "See the failing step"

    @property
    def agent_label(self) -> str:
        return self.agent

    @property
    def tag(self) -> str | None:
        return f"{self.consecutive_failures} fails on one step"


class BlockFailedCause(BaseCause):
    cause: Literal[AlertCause.BLOCK_FAILED] = AlertCause.BLOCK_FAILED
    agent: str
    step: str
    failure_count: int
    error: str

    @property
    def headline(self) -> str:
        return f"{self.agent} keeps failing on one step"

    @property
    def body(self) -> str:
        return (
            f"Every run gets as far as “{self.step}” and stops there. The rest of "
            f"{self.agent} is fine — this one step needs a look."
        )

    @property
    def cta_label(self) -> str:
        return "See the failing step"

    @property
    def agent_label(self) -> str:
        return self.agent

    @property
    def tag(self) -> str | None:
        return f"{self.failure_count} failures"

    @property
    def facts(self) -> list[AlertFact]:
        return [
            AlertFact(label="Failing step", value=self.step),
            AlertFact(label="Failures", value=str(self.failure_count)),
            AlertFact(label="Error", value=self.error),
        ]


class ContinuousErrorCause(BaseCause):
    cause: Literal[AlertCause.CONTINUOUS_ERROR] = AlertCause.CONTINUOUS_ERROR
    agent: str
    days: int
    failing_since_label: str
    consecutive_failures: int
    error: str
    credits_spent: float
    fix_label: str = "See what’s failing"

    @property
    def headline(self) -> str:
        day_word = "day" if self.days == 1 else "days"
        return f"{self.agent} has been failing for {self.days} {day_word}"

    @property
    def body(self) -> str:
        return (
            f"Every scheduled run since {self.failing_since_label} has ended the same "
            "way, and each attempt still costs credits."
        )

    @property
    def cta_label(self) -> str:
        return self.fix_label

    @property
    def agent_label(self) -> str:
        return self.agent

    @property
    def tag(self) -> str | None:
        return f"failing {self.days}d"

    @property
    def facts(self) -> list[AlertFact]:
        return [
            AlertFact(label="Failing since", value=self.failing_since_label),
            AlertFact(label="Consecutive failures", value=str(self.consecutive_failures)),
            AlertFact(label="Error", value=self.error),
            AlertFact(
                label="Credits spent retrying", value=f"{self.credits_spent:,.2f}"
            ),
        ]


class AwaitingReviewCause(BaseCause):
    cause: Literal[AlertCause.AWAITING_REVIEW] = AlertCause.AWAITING_REVIEW
    agent: str
    count: int
    since_label: str

    @property
    def headline(self) -> str:
        return f"{self.agent} is waiting on your review"

    @property
    def body(self) -> str:
        output = "1 output" if self.count == 1 else f"{self.count} outputs"
        return (
            f"{output} waiting since {self.since_label}. Nothing sends until you "
            "approve or dismiss."
        )

    @property
    def cta_label(self) -> str:
        return f"Review {self.count} output" + ("" if self.count == 1 else "s")

    @property
    def agent_label(self) -> str:
        return self.agent

    @property
    def tag(self) -> str | None:
        return f"{self.count} waiting"


class AwaitingInputCause(BaseCause):
    cause: Literal[AlertCause.AWAITING_INPUT] = AlertCause.AWAITING_INPUT
    agent: str
    field_name: str

    @property
    def headline(self) -> str:
        return f"{self.agent} needs an input from you"

    @property
    def body(self) -> str:
        return f"Its current run is waiting for a value for “{self.field_name}”."

    @property
    def cta_label(self) -> str:
        return "Provide the input"

    @property
    def agent_label(self) -> str:
        return self.agent

    @property
    def tag(self) -> str | None:
        return "waiting on input"


class LowBalanceCause(BaseCause):
    cause: Literal[AlertCause.LOW_BALANCE] = AlertCause.LOW_BALANCE
    days_left: int
    daily_rate_display: str
    balance_display: str
    runs_out_label: str
    scheduled_agents: int

    @property
    def headline(self) -> str:
        return f"Your credits run out in about {self.days_left} days"

    @property
    def body(self) -> str:
        agents = (
            "1 scheduled agent would stop"
            if self.scheduled_agents == 1
            else f"{self.scheduled_agents} scheduled agents would stop"
        )
        return (
            f"At {self.daily_rate_display}/day the {self.balance_display} remaining "
            f"run out around {self.runs_out_label}. {agents}."
        )

    @property
    def cta_label(self) -> str:
        return "Top up credits"

    @property
    def agent_label(self) -> str:
        return "Your balance"

    @property
    def tag(self) -> str | None:
        return f"~{self.days_left} days left"


class ZeroBalanceCause(BaseCause):
    cause: Literal[AlertCause.ZERO_BALANCE] = AlertCause.ZERO_BALANCE
    agent: str
    shortfall_display: str

    @property
    def headline(self) -> str:
        return f"{self.agent} stopped — you are out of credits"

    @property
    def body(self) -> str:
        return (
            f"{self.agent} needs {self.shortfall_display} more credits to finish. It "
            "stays stopped until you top up."
        )

    @property
    def cta_label(self) -> str:
        return "Top up credits"

    @property
    def agent_label(self) -> str:
        return self.agent

    @property
    def tag(self) -> str | None:
        return "out of credits"


class GuardrailCause(BaseCause):
    cause: Literal[AlertCause.GUARDRAIL] = AlertCause.GUARDRAIL
    agent: str
    limit_display: str
    period_noun: str
    reset_label: str

    @property
    def headline(self) -> str:
        return f"{self.agent} hit its spend limit"

    @property
    def body(self) -> str:
        return (
            f"Reached its {self.limit_display}-credit limit for this "
            f"{self.period_noun} and stopped mid-run. Resumes {self.reset_label}."
        )

    @property
    def cta_label(self) -> str:
        return "Review spend limit"

    @property
    def agent_label(self) -> str:
        return self.agent

    @property
    def tag(self) -> str | None:
        return "spend limit hit"


AlertCausePayload = Annotated[
    Union[
        AuthExpiredCause,
        PausedFailuresCause,
        BlockFailedCause,
        ContinuousErrorCause,
        AwaitingReviewCause,
        AwaitingInputCause,
        LowBalanceCause,
        ZeroBalanceCause,
        GuardrailCause,
    ],
    Field(discriminator="cause"),
]

_BY_CAUSE: dict[AlertCause, type[BaseCause]] = {
    AlertCause.AUTH_EXPIRED: AuthExpiredCause,
    AlertCause.PAUSED_FAILURES: PausedFailuresCause,
    AlertCause.BLOCK_FAILED: BlockFailedCause,
    AlertCause.CONTINUOUS_ERROR: ContinuousErrorCause,
    AlertCause.AWAITING_REVIEW: AwaitingReviewCause,
    AlertCause.AWAITING_INPUT: AwaitingInputCause,
    AlertCause.LOW_BALANCE: LowBalanceCause,
    AlertCause.ZERO_BALANCE: ZeroBalanceCause,
    AlertCause.GUARDRAIL: GuardrailCause,
}


def parse_cause(cause: AlertCause, data: dict) -> BaseCause:
    """Rebuild a stored condition into its cause model."""
    return _BY_CAUSE[cause].model_validate(data)
