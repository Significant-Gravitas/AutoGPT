"""How a user lets Otto hand work to their team.

Stored per user on ``User.delegationSettings``; a user who never saved them
gets these defaults. ``mode`` is the approval mode a thread Otto delegates
starts in; the caps are enforced by ``delegate_to_expert``.
"""

from pydantic import BaseModel, Field

from backend.copilot.model import AutopilotMode

# The gate's own default (``gate.policy.DEFAULT_MODE``), repeated rather than
# imported: this model is read by the DatabaseManager, which must not load
# the gate package. ``delegation_settings_test`` pins the two together.
DEFAULT_DELEGATION_MODE: AutopilotMode = "auto"
MAX_CAP_USD = 1_000.0
MAX_DAILY_BUDGET_USD = 10_000.0


class DelegationSettings(BaseModel):
    mode: AutopilotMode = DEFAULT_DELEGATION_MODE
    per_delegation_cap_usd: float = Field(default=2.0, ge=0, le=MAX_CAP_USD)
    daily_budget_usd: float = Field(default=10.0, ge=0, le=MAX_DAILY_BUDGET_USD)
    ask_before_external: bool = True
    ask_before_over_cap: bool = True
    new_experts_ask_first: bool = False


class DelegationSettingsUpdate(BaseModel):
    """The whole settings object, as the Settings tab saves it.

    Every field is required, unlike :class:`DelegationSettings`: a model with
    defaults gets separate input and output schemas, which the generated
    client cannot name.
    """

    mode: AutopilotMode
    per_delegation_cap_usd: float = Field(ge=0, le=MAX_CAP_USD)
    daily_budget_usd: float = Field(ge=0, le=MAX_DAILY_BUDGET_USD)
    ask_before_external: bool
    ask_before_over_cap: bool
    new_experts_ask_first: bool
