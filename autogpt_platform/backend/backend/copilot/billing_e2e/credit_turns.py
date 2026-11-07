"""SDK turns billed to the credit balance once the plan's allowance is used up.

Each turn is the AutoPilot block's, wrapped as ``CoPilotProcessor._run_turn``
wraps such a turn: ``bill_to_credits`` puts it on the balance, and the engine
runs inside ``credit_billing``, which charges each cost as it lands and the
rest when the turn ends. The processor's other work around the turn (the
allowance lookup, the stream registry, the turn's lock) is not repeated here;
``processor_budget_gate_test`` checks that wiring.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from backend.copilot import credit_billing as billing
from backend.copilot.config import ChatConfig
from backend.copilot.constants import AUTOPILOT_BLOCK_TOOL_NAME
from backend.copilot.model import ChatSession
from backend.copilot.turn_pause import TurnPause

from .provider import FakeProvider
from .sdk_turns import SdkTurns

# What ``refusal_for`` answers once the plan's daily allowance is used up.
_ALLOWANCE_USED_UP = TurnPause(
    reason="daily",
    resets_at=datetime(2026, 10, 17, tzinfo=UTC),
    checkpoint_saved=False,
    started=False,
)


class CreditBilledTurns(SdkTurns):
    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        provider: FakeProvider,
        config: ChatConfig,
        *,
        cli_max_retries: int | None = None,
    ) -> None:
        super().__init__(
            tmp_path, monkeypatch, provider, config, cli_max_retries=cli_max_retries
        )
        # A deployment that bills credits, with the per-turn ceiling and the
        # transport its SDK engine runs on.
        monkeypatch.setattr(billing.settings.config, "enable_credit", True)
        monkeypatch.setattr(billing, "config", config)

    async def __call__(
        self, session: ChatSession, prompt: str, *, new_pod: bool
    ) -> None:
        turn = await billing.bill_to_credits(
            user_id=session.user_id,
            session_id=session.session_id,
            tool_name=AUTOPILOT_BLOCK_TOOL_NAME,
            refusal=_ALLOWANCE_USED_UP,
        )
        assert turn is not None, "the balance didn't cover the turn, so it was refused"
        async with billing.credit_billing(turn):
            await super().__call__(session, prompt, new_pod=new_pod)
