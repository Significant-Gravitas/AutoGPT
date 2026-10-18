"""Every route a chat turn can take, and what each should charge per call.

A row names the engine, how the deployment is configured, and the basis its
charge is expected to rest on:

- ``invoice``: what the provider billed for the call.
- ``rate_card``: Anthropic's list price for the tokens, which the platform
  charges where it has no invoice to read. Never the CLI's own estimate.
- ``moonshot_estimate``: the platform's Kimi rate card, the fallback for a
  Kimi call with no invoice.
- ``none``: the user's own subscription pays, or nobody does; the platform
  must not charge the user's spend budgets at all.

A turn the platform relaunches after a provider failure owes only the calls
billed after the failure. The failed attempt is the platform's to absorb, not
the user's. A turn the failure ends, because it had already streamed a tool
call that a relaunch would roll back and could repeat, owes nothing.
"""

from typing import Any, Literal

from pydantic import BaseModel

from .invoice import ANTHROPIC_PRICES, KIMI, MOONSHOT_FALLBACK, SONNET, Call
from .provider import PlannedFault

ChargeBasis = Literal["invoice", "rate_card", "moonshot_estimate", "none"]
# What a finding is about: whether the thread took its planned path, what the
# turn was charged, the tokens recorded against it, or the live spend the SDK
# turn's governor lands tasks on.
Check = Literal["path", "charge", "tokens", "meter"]


class KnownGap(BaseModel):
    """A known defect: this row's ``check`` fails until it is fixed.

    The other checks still hold the row to account, and a gap that stops
    reproducing fails the test so it gets removed.
    """

    check: Check
    reason: str


class Scenario(BaseModel):
    id: str
    engine: Literal["sdk", "microsoft_365"]
    model: str
    charge: ChargeBasis
    cost_log_provider: str
    # ChatConfig fields; ``{provider}`` in a string is the fake's base URL.
    config: dict[str, Any] = {}
    codex: bool = False
    generation_lookups_fail: bool = False
    # Calls the provider bills per turn: one, a tool round of two, one.
    calls_per_turn: tuple[int, int, int] = (1, 2, 1)
    fault: PlannedFault | None = None
    # Whether the platform relaunches the turn after the fault, rather than
    # its client retrying the one request. The fault then ends the failed
    # attempt, so every call after it is the retry's: the CLI's own retries
    # must be off (``cli_max_retries=0``) for this to hold.
    fault_relaunches_turn: bool = False
    # Whether the fault ends the turn in an error instead: the turn had
    # already streamed a tool call, which a relaunch would roll back out of
    # the chat and could run again.
    fault_ends_turn: bool = False
    # CLAUDE_CODE_MAX_RETRIES for the CLI; None keeps its own default.
    cli_max_retries: int | None = None
    known_gap: KnownGap | None = None

    def owed_calls(self, turn: int, calls: list[Call]) -> list[Call]:
        """The turn's billed calls the user pays for: after a relaunch, only
        the attempt that succeeded, and none for a turn the fault ended."""
        fault = self.fault
        if fault is None or fault.turn != turn:
            return calls
        if self.fault_ends_turn:
            return []
        if not self.fault_relaunches_turn:
            return calls
        return [call for call in calls if call.request > fault.request]

    @property
    def needs_cli(self) -> bool:
        return self.engine == "sdk"

    @property
    def through_gateway(self) -> bool:
        """Whether the CLI reaches the provider through the model gateway, so
        the provider sees the gateway's requests rather than the CLI's."""
        return self.codex or self.config.get("platform_route") == "local"

    @property
    def reports_usage(self) -> bool:
        """Whether the upstream reports token counts the platform logs as is."""
        return self.engine != "microsoft_365"


_OPENROUTER = {
    "platform_route": "openrouter",
    "openrouter_api_key": "sk-or-billing-e2e",
    "openrouter_base_url": "{provider}/openrouter/api",
}
# The fake provider never checks the key.
_DIRECT = {
    "platform_route": "direct_anthropic",
    "anthropic_api_key": "sk-ant-billing-e2e",
    "anthropic_base_url": "{provider}/anthropic",
}

_SDK_ANTHROPIC = Scenario(
    id="sdk-anthropic",
    engine="sdk",
    model=SONNET,
    charge="invoice",
    cost_log_provider="anthropic",
    config={**_DIRECT, "standard_model": SONNET},
)


def _sdk_anthropic_failing(
    scenario_id: str,
    fault: PlannedFault,
    calls_per_turn: tuple[int, int, int],
    *,
    relaunches: bool = True,
) -> Scenario:
    """The direct-Anthropic thread with one request failed and the CLI's own
    retries off, so the platform's handling of the failure is what runs."""
    return _SDK_ANTHROPIC.model_copy(
        update={
            "id": scenario_id,
            "fault": fault,
            "fault_relaunches_turn": relaunches,
            "fault_ends_turn": not relaunches,
            "calls_per_turn": calls_per_turn,
            "cli_max_retries": 0,
        }
    )


SCENARIOS = [
    _SDK_ANTHROPIC,
    # The tool round's second call is refused after the tool ran. Relaunching
    # would roll the call the user watched out of the chat and could repeat
    # it, so the turn ends in the error: the provider billed the tool round's
    # first call, and a turn that failed owes nothing.
    _sdk_anthropic_failing(
        "sdk-anthropic-overloaded-after-tool",
        PlannedFault(kind="overloaded", turn=1, request=2),
        calls_per_turn=(1, 1, 1),
        relaunches=False,
    ),
    # The connection drops instead, which is transient too, and ends the turn
    # the same way.
    _sdk_anthropic_failing(
        "sdk-anthropic-dropped-after-tool",
        PlannedFault(kind="dropped", turn=1, request=2),
        calls_per_turn=(1, 1, 1),
        relaunches=False,
    ),
    # A chat's first request drops before anything streamed. The relaunch
    # starts the CLI session over, which the CLI refuses while the failed
    # launch's session file is still there, and is charged its one call.
    _sdk_anthropic_failing(
        "sdk-anthropic-dropped-first-turn",
        PlannedFault(kind="dropped", turn=0, request=1),
        calls_per_turn=(1, 2, 1),
    ),
    # A resumed turn whose first request is refused is relaunched on the
    # session before anything streamed, and charged its one successful call.
    _sdk_anthropic_failing(
        "sdk-anthropic-overloaded-on-resume",
        PlannedFault(kind="overloaded", turn=2, request=1),
        calls_per_turn=(1, 2, 1),
    ),
    Scenario(
        id="sdk-openrouter",
        engine="sdk",
        model=f"anthropic/{SONNET}",
        charge="invoice",
        cost_log_provider="open_router",
        config={**_OPENROUTER, "standard_model": f"anthropic/{SONNET}"},
    ),
    # OpenRouter refuses the bill lookups: the turn is charged the
    # platform's rate card, not the CLI's estimate.
    Scenario(
        id="sdk-openrouter-lookups-refused",
        engine="sdk",
        model=f"anthropic/{SONNET}",
        charge="rate_card",
        cost_log_provider="open_router",
        config={**_OPENROUTER, "standard_model": f"anthropic/{SONNET}"},
        generation_lookups_fail=True,
    ),
    Scenario(
        id="sdk-openrouter-kimi",
        engine="sdk",
        model=KIMI,
        charge="invoice",
        cost_log_provider="open_router",
        config={**_OPENROUTER, "standard_model": KIMI},
    ),
    Scenario(
        id="sdk-openrouter-kimi-lookups-refused",
        engine="sdk",
        model=KIMI,
        charge="moonshot_estimate",
        cost_log_provider="open_router",
        config={**_OPENROUTER, "standard_model": KIMI},
        generation_lookups_fail=True,
    ),
    Scenario(
        id="sdk-codex",
        engine="sdk",
        model="",
        charge="none",
        cost_log_provider="codex",
        config=_DIRECT,
        codex=True,
    ),
    # The local route runs on the SDK engine through the shared model
    # gateway: the operator's server is never charged, through either adapter.
    Scenario(
        id="sdk-local-openai-chat",
        engine="sdk",
        model="llama3.1:8b",
        charge="none",
        cost_log_provider="local",
        config={
            "platform_route": "local",
            "local_base_url": "{provider}/ollama/v1",
            "local_adapter": "openai_chat",
            "local_context_tokens": 131_072,
            "standard_model": "llama3.1:8b",
            "advanced_model": "llama3.1:8b",
        },
    ),
    Scenario(
        id="sdk-local-anthropic-passthrough",
        engine="sdk",
        model="llama3.1:8b",
        charge="none",
        cost_log_provider="local",
        config={
            "platform_route": "local",
            "local_base_url": "{provider}/ollama",
            "local_adapter": "anthropic_passthrough",
            "local_context_tokens": 131_072,
            "standard_model": "llama3.1:8b",
            "advanced_model": "llama3.1:8b",
        },
    ),
    Scenario(
        id="microsoft-365-copilot",
        engine="microsoft_365",
        model="microsoft-365-copilot",
        charge="none",
        cost_log_provider="microsoft_365_copilot",
        # Graph runs no tools, so the tool turn is one call too.
        calls_per_turn=(1, 1, 1),
    ),
]


def expected_charge_usd(basis: ChargeBasis, call: Call) -> float:
    if basis == "invoice":
        return call.invoice_usd
    if basis == "rate_card":
        list_price = ANTHROPIC_PRICES[call.model.removeprefix("anthropic/")]
        return list_price.of(call.usage, call.cache_ttl)
    if basis == "moonshot_estimate":
        return MOONSHOT_FALLBACK.of(call.usage, None)
    return 0.0
