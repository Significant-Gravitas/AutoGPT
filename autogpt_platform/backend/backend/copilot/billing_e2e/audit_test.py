"""The audit on its own: a relaunched turn owes only the attempt that
succeeded, a turn the failure ended owes nothing, a turn billed to credits is
debited as its cost lands, and a governor meters each call at list price."""

from claude_agent_sdk import AssistantMessage, TextBlock

from backend.copilot.model import Usage
from backend.copilot.sdk.usage_meter import UsageMeter
from backend.data.credit import UsageTransactionMetadata
from backend.data.platform_cost import PlatformCostEntry, usd_to_microdollars

from .audit import _audit_meter, _audit_turn
from .invoice import ANTHROPIC_PRICES, SONNET, Call, Price, scripted_usage
from .ledger import (
    CostLogRow,
    CreditBilling,
    CreditDebit,
    PlannedDrain,
    PlatformLedger,
    SpendCharge,
    TreeCharge,
)
from .provider import PlannedFault
from .scenarios import SCENARIOS, _sdk_anthropic_failing, expected_charge_usd

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


_CREDIT_BILLED = next(s for s in SCENARIOS if s.id == "sdk-anthropic-credits")
# 42,127 microdollars: 4 whole credits and part of a fifth.
_TOOL_ROUND = [_call(1, request=1), _call(2, request=2)]


def test_a_turn_debited_its_whole_credits_then_the_part_credit_passes():
    ledger, usage = _credit_billed_turn(debits=[4, 1])

    assert _audit_turn(_CREDIT_BILLED, 1, _TOOL_ROUND, ledger, usage) == []


def test_a_turn_debited_its_whole_cost_at_once_is_reported():
    """What a charge that rounds up as the cost lands takes, and one that
    took the cost after the turn had ended."""
    ledger, usage = _credit_billed_turn(debits=[5])

    findings = _audit_turn(_CREDIT_BILLED, 1, _TOOL_ROUND, ledger, usage)

    assert [finding.check for finding in findings] == ["charge"]
    assert "owes [4, 1]" in findings[0].message


def test_a_turn_billed_to_credits_charging_the_spend_windows_is_reported():
    ledger, usage = _credit_billed_turn(debits=[4, 1])
    ledger.spend.append(
        SpendCharge(turn=1, user_id="user", microdollars=42_127, skip_daily=False)
    )

    findings = _audit_turn(_CREDIT_BILLED, 1, _TOOL_ROUND, ledger, usage)

    assert [finding.check for finding in findings] == ["charge"]
    assert "spend windows" in findings[0].message


def test_a_turn_the_balance_left_a_credit_owes_that_credit():
    drained = _CREDIT_BILLED.model_copy(
        update={"credits": CreditBilling(drain=PlannedDrain(turn=1, left=1))}
    )
    ledger, usage = _credit_billed_turn(debits=[1])

    assert _audit_turn(drained, 1, _TOOL_ROUND, ledger, usage) == []


def _credit_billed_turn(debits: list[int]) -> tuple[PlatformLedger, list[Usage]]:
    """Turn 1's ledger and usage, as if the platform billed ``_TOOL_ROUND``
    to credits in ``debits``."""
    ledger, usage = _turn_charged_for(*_TOOL_ROUND)
    ledger.spend.clear()
    ledger.cost_log[0].entry.metadata = {"billing_mode": "credits"}
    ledger.debits = [
        CreditDebit(
            turn=1, user_id="user", credits=debit, metadata=UsageTransactionMetadata()
        )
        for debit in debits
    ]
    return ledger, usage


# Two calls on OpenRouter, which names the model ``anthropic/claude-sonnet-5-5``.
_OPENROUTER_CALLS = [
    call.model_copy(update={"route": "openrouter", "model": f"anthropic/{SONNET}"})
    for call in (_FAILED_ATTEMPT, _RETRY)
]


def test_a_governor_metering_its_calls_at_list_price_passes():
    meter = UsageMeter()
    for call in _OPENROUTER_CALLS:
        meter.observe(_as_the_cli_reports(call))

    assert _audit_meter(1, _OPENROUTER_CALLS, meter.spent_usd) == []


def test_a_governor_pricing_sonnet_as_opus_is_reported():
    opus = Price(
        input=15, output=75, cache_read=1.5, cache_write_5m=18.75, cache_write_1h=30
    )
    metered = sum(opus.of(call.usage, "1h") for call in _OPENROUTER_CALLS)

    findings = _audit_meter(1, _OPENROUTER_CALLS, metered)

    assert [finding.check for finding in findings] == ["meter"]


def _as_the_cli_reports(call: Call) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text="x")],
        model=call.model,
        usage={
            "input_tokens": call.usage.input_tokens,
            "output_tokens": call.usage.output_tokens,
            "cache_read_input_tokens": call.usage.cache_read_tokens,
            "cache_creation_input_tokens": call.usage.cache_write_tokens,
        },
        message_id=f"call-{call.request}",
    )


def _turn_charged_for(*calls: Call) -> tuple[PlatformLedger, list[Usage]]:
    """Turn 1's ledger and session usage, as if the platform charged ``calls``
    on the direct route's basis."""
    microdollars = usd_to_microdollars(
        sum(expected_charge_usd(_RELAUNCHED.charge, call) for call in calls)
    )
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
