"""A finished thread checked against its provider's invoice, turn by turn."""

from pydantic import BaseModel

from backend.copilot.model import ChatSession, Usage
from backend.data.platform_cost import PlatformCostEntry, usd_to_microdollars

from .invoice import Call
from .ledger import PlatformLedger
from .provider import FakeProvider
from .scenarios import Check, Scenario, expected_charge_usd

# Per call: float pricing in the CLI and in each sum, rounded once per turn.
_MICRODOLLAR_TOLERANCE_PER_CALL = 2


class Finding(BaseModel):
    check: Check
    message: str


def audit_thread(
    scenario: Scenario,
    provider: FakeProvider,
    ledger: PlatformLedger,
    session: ChatSession,
    usage_by_turn: list[list[Usage]],
    cli_resumed: list[tuple[int, bool]] | None,
) -> list[Finding]:
    """Every way the thread strayed from its invoice or its planned path.

    ``cli_resumed`` says, for each CLI launch of an SDK thread, its turn and
    whether it resumed the chat's session; None for engines without a CLI.
    """
    findings = _audit_path(provider, cli_resumed, turns=len(usage_by_turn))
    findings += _audit_attribution(session, ledger)
    for turn, session_usage in enumerate(usage_by_turn):
        findings += _audit_turn(
            scenario, turn, provider.calls_in(turn), ledger, session_usage
        )
    return findings


def _audit_attribution(session: ChatSession, ledger: PlatformLedger) -> list[Finding]:
    """Every charge on this chat's user and its daily budget, every row on
    this chat."""
    messages = [
        f"turn {charge.turn}: charged user {charge.user_id!r}, "
        f"skipping the daily budget: {charge.skip_daily}"
        for charge in ledger.spend
        if charge.user_id != session.user_id or charge.skip_daily
    ]
    messages += [
        f"turn {row.turn}: cost-log row for user {row.entry.user_id!r}, "
        f"chat {row.entry.chat_session_id!r}"
        for row in ledger.cost_log
        if (row.entry.user_id, row.entry.chat_session_id)
        != (session.user_id, session.session_id)
    ]
    return [Finding(check="charge", message=message) for message in messages]


def _audit_path(
    provider: FakeProvider,
    cli_resumed: list[tuple[int, bool]] | None,
    *,
    turns: int,
) -> list[Finding]:
    """Whether the thread took the path it meant to.

    Without this the charge checks could pass vacuously: an SDK turn that
    never resumed has no restored total to over-charge from, and a fault that
    never fired tests nothing.
    """
    messages = [f"unexpected provider request: {r}" for r in provider.unexpected]
    if provider.faults_served != provider.faults:
        messages.append(
            f"faults served {provider.faults_served}, planned {provider.faults}"
        )
    # The first turn's launches start the session, retries included; every
    # later turn's launches resume it.
    if cli_resumed is not None and (
        {turn for turn, _ in cli_resumed} != set(range(turns))
        or any(resumed != (turn > 0) for turn, resumed in cli_resumed)
    ):
        messages.append(f"CLI launches (turn, resumed): {cli_resumed}")
    return [Finding(check="path", message=message) for message in messages]


def _audit_turn(
    scenario: Scenario,
    turn: int,
    calls: list[Call],
    ledger: PlatformLedger,
    session_usage: list[Usage],
) -> list[Finding]:
    if len(calls) != scenario.calls_per_turn[turn]:
        # The turn did not run as planned; checking its charge would only
        # compare against the wrong invoice.
        return [
            Finding(
                check="path",
                message=f"turn {turn}: the provider billed {len(calls)} calls",
            )
        ]
    owed = scenario.owed_calls(turn, calls)
    if not owed:
        return _audit_unbilled(turn, ledger, session_usage)
    findings = _audit_spend(scenario, turn, owed, ledger)
    rows = ledger.rows_in(turn)
    if len(rows) != 1:
        message = f"turn {turn}: {len(rows)} cost-log rows, expected 1"
        return [*findings, Finding(check="charge", message=message)]
    findings += _audit_row(scenario, turn, rows[0], ledger)
    if scenario.reports_usage:
        findings += _audit_tokens(turn, owed, rows[0], session_usage)
    return findings


def _audit_unbilled(
    turn: int, ledger: PlatformLedger, session_usage: list[Usage]
) -> list[Finding]:
    """A turn that owes nothing leaves no charge, cost-log row or usage."""
    spend = [charge.microdollars for charge in ledger.spend_in(turn)]
    tree = [charge.microdollars for charge in ledger.tree_in(turn)]
    rows = ledger.rows_in(turn)
    if not (spend or tree or rows or session_usage):
        return []
    message = (
        f"turn {turn} owes nothing, yet charged {spend} to spend and {tree} to "
        f"the tree, logged {len(rows)} cost-log row(s) and recorded "
        f"{len(session_usage)} session usage entries"
    )
    return [Finding(check="charge", message=message)]


def _audit_spend(
    scenario: Scenario, turn: int, calls: list[Call], ledger: PlatformLedger
) -> list[Finding]:
    spend = [charge.microdollars for charge in ledger.spend_in(turn)]
    tree = [charge.microdollars for charge in ledger.tree_in(turn)]
    if scenario.charge == "none":
        if not spend and not tree:
            return []
        message = (
            f"turn {turn}: charged {spend} to spend and {tree} to the tree; "
            f"nobody bills the platform for this route"
        )
        return [Finding(check="charge", message=message)]
    expected = usd_to_microdollars(
        sum(expected_charge_usd(scenario.charge, call) for call in calls)
    )
    tolerance = _MICRODOLLAR_TOLERANCE_PER_CALL * len(calls)
    messages = []
    if len(spend) != 1 or abs(spend[0] - (expected or 0)) > tolerance:
        messages.append(
            f"turn {turn}: charged {spend} microdollars; the turn owes "
            f"{expected} ({scenario.charge}) for {len(calls)} call(s)"
        )
    if tree != spend:
        messages.append(f"turn {turn}: the tree was charged {tree}, spend {spend}")
    return [Finding(check="charge", message=message) for message in messages]


def _audit_row(
    scenario: Scenario, turn: int, row: PlatformCostEntry, ledger: PlatformLedger
) -> list[Finding]:
    messages = []
    if row.provider != scenario.cost_log_provider:
        messages.append(f"turn {turn}: cost-log provider {row.provider!r}")
    if scenario.charge == "none":
        if row.tracking_type != "tokens" or row.cost_microdollars:
            messages.append(
                f"turn {turn}: logged {row.tracking_type} {row.cost_microdollars} "
                f"for a route the platform does not pay for"
            )
    else:
        charged = [charge.microdollars for charge in ledger.spend_in(turn)]
        if row.tracking_type != "cost_usd" or [row.cost_microdollars] != charged:
            messages.append(
                f"turn {turn}: logged {row.tracking_type} {row.cost_microdollars}, "
                f"charged {charged}"
            )
    return [Finding(check="charge", message=message) for message in messages]


def _audit_tokens(
    turn: int, calls: list[Call], row: PlatformCostEntry, session_usage: list[Usage]
) -> list[Finding]:
    owed = (
        sum(call.usage.input_tokens for call in calls),
        sum(call.usage.output_tokens for call in calls),
        sum(call.usage.cache_read_tokens for call in calls),
        sum(call.usage.cache_write_tokens for call in calls),
    )
    logged = (
        row.input_tokens or 0,
        row.output_tokens or 0,
        row.cache_read_tokens or 0,
        row.cache_creation_tokens or 0,
    )
    recorded = [
        (
            u.prompt_tokens,
            u.completion_tokens,
            u.cache_read_tokens,
            u.cache_creation_tokens,
        )
        for u in session_usage
    ]
    messages = []
    if logged != owed:
        messages.append(f"turn {turn}: logged tokens {logged}, the turn owes {owed}")
    if recorded != [owed]:
        messages.append(f"turn {turn}: session usage {recorded}, the turn owes {owed}")
    return [Finding(check="tokens", message=message) for message in messages]
