"""The audit on its own: a relaunched turn owes only the attempt that
succeeded, and a turn the failure ended owes nothing."""

from backend.copilot.model import Usage
from backend.data.platform_cost import PlatformCostEntry, usd_to_microdollars

from .audit import _audit_turn
from .invoice import ANTHROPIC_PRICES, SONNET, Call, scripted_usage
from .ledger import CostLogRow, PlatformLedger, SpendCharge, TreeCharge
from .provider import PlannedFault
from .scenarios import SCENARIOS, _sdk_anthropic_failing

# A turn relaunched after the provider billed a call that streamed nothing.
_RELAUNCHED = _sdk_anthropic_failing(
    "relaunched-after-a-billed-call",
    PlannedFault(kind="dropped", turn=1, request=2),
    calls_per_turn=(1, 2, 1),
)
_ENDED = next(s for s in SCENARIOS if s.id == "sdk-anthropic-dropped-after-tool")


def _call(index: int, request: int) -> Call:
    usage = scripted_usage(index, cached=True)
    return Call(
        turn=1,
        request=request,
        route="anthropic",
        model=SONNET,
        usage=usage,
        cache_ttl="5m",
        invoice_usd=ANTHROPIC_PRICES[SONNET].of(usage, "5m"),
    )


# Turn 1's first call, then the dropped request 2, then a relaunch's call.
_FAILED_ATTEMPT = _call(1, request=1)
_RETRY = _call(2, request=3)


def test_a_relaunched_turn_charged_only_its_retry_passes():
    ledger, usage = _turn_charged_for(_RETRY)

    findings = _audit_turn(_RELAUNCHED, 1, [_FAILED_ATTEMPT, _RETRY], ledger, usage)

    assert findings == []


def test_charging_the_failed_attempt_is_reported():
    ledger, usage = _turn_charged_for(_FAILED_ATTEMPT, _RETRY)

    findings = _audit_turn(_RELAUNCHED, 1, [_FAILED_ATTEMPT, _RETRY], ledger, usage)

    assert sorted(finding.check for finding in findings) == [
        "charge",
        "tokens",
        "tokens",
    ]
    assert all("the turn owes" in finding.message for finding in findings)


def test_a_turn_the_failure_ended_charged_nothing_passes():
    findings = _audit_turn(_ENDED, 1, [_FAILED_ATTEMPT], PlatformLedger(), [])

    assert findings == []


def test_charging_a_turn_the_failure_ended_is_reported():
    ledger, usage = _turn_charged_for(_FAILED_ATTEMPT)

    findings = _audit_turn(_ENDED, 1, [_FAILED_ATTEMPT], ledger, usage)

    assert [finding.check for finding in findings] == ["charge"]
    assert "owes nothing" in findings[0].message


def _turn_charged_for(*calls: Call) -> tuple[PlatformLedger, list[Usage]]:
    """Turn 1's ledger and session usage, as if the platform charged ``calls``."""
    microdollars = usd_to_microdollars(sum(call.invoice_usd for call in calls))
    input_tokens = sum(call.usage.input_tokens for call in calls)
    output_tokens = sum(call.usage.output_tokens for call in calls)
    cache_read = sum(call.usage.cache_read_tokens for call in calls)
    cache_write = sum(call.usage.cache_write_tokens for call in calls)
    ledger = PlatformLedger()
    ledger.spend.append(
        SpendCharge(turn=1, user_id="user", microdollars=microdollars, skip_daily=False)
    )
    ledger.tree.append(TreeCharge(turn=1, tree_id="tree", microdollars=microdollars))
    ledger.cost_log.append(
        CostLogRow(
            turn=1,
            entry=PlatformCostEntry(
                user_id="user",
                chat_session_id="chat",
                provider="anthropic",
                cost_microdollars=microdollars,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_write,
                tracking_type="cost_usd",
            ),
        )
    )
    usage = Usage(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_write,
    )
    return ledger, [usage]
