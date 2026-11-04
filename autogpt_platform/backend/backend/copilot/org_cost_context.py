"""The usage attribution of work that runs in an organization.

``trial_cost_context`` carries the attribution (which trial and which spend
counters a piece of work charges) and captures a person's own. Work that runs
in a team organization counts in the person's own counters there, apart from
their personal ones and from other organizations', and never toward their
personal trial: that attribution is captured here, through the entitlement
check that knows which organizations are team ones. ``rate_limit`` imports
``trial_cost_context``, so that module can't ask the check itself.
"""

from collections.abc import Awaitable, Callable
from functools import wraps
from inspect import signature
from typing import ParamSpec, TypeVar

from pydantic import TypeAdapter

from backend.copilot.trial_cost_context import (
    TrialCostContext,
    capture_cost_context,
    get_trial_cost_context,
    trial_cost_context,
)
from backend.data.entitlement_limits import team_org_id_for

P = ParamSpec("P")
R = TypeVar("R")

# The parameters a function given to ``attributed_usage`` names its
# organization by.
_ORG_PARAMETERS = ("org_id", "organization_id")


async def capture_org_cost_context(
    user_id: str | None, org_id: str | None
) -> TrialCostContext:
    """The usage attribution of work *user_id* starts now in *org_id*.

    The attribution already in force wins, as in ``capture_cost_context``: a
    turn's tools and children keep the one the turn started under. In a team
    organization the usage counts in the person's own counters there and
    never toward their personal trial; elsewhere it is their own attribution.
    """
    if user_id is None or get_trial_cost_context(user_id) is not None:
        return await capture_cost_context(user_id)
    team = await team_cost_context(user_id, org_id)
    return team or await capture_cost_context(user_id)


async def team_cost_context(
    user_id: str, org_id: str | None
) -> TrialCostContext | None:
    """The attribution of usage *user_id* spends in *org_id* when that is a
    team organization: their counters there, and no personal trial. None for
    a personal organization, or none."""
    team_org_id = await team_org_id_for(org_id)
    if team_org_id is None:
        return None
    return TrialCostContext(user_id=user_id, trial_id=None, team_org_id=team_org_id)


def attributed_usage(function: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """Capture standalone/background work before its first provider request.

    The work runs for its ``user_id`` argument in the organization its
    ``org_id`` or ``organization_id`` argument names; without one, or with
    None, in the person's own.
    """
    parameters = signature(function)
    org_parameter = next(
        (name for name in _ORG_PARAMETERS if name in parameters.parameters), None
    )
    optional_id = TypeAdapter(str | None)

    @wraps(function)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        arguments = parameters.bind(*args, **kwargs)
        arguments.apply_defaults()
        user_id = optional_id.validate_python(arguments.arguments["user_id"])
        org_id = (
            optional_id.validate_python(arguments.arguments[org_parameter])
            if org_parameter is not None
            else None
        )
        context = await capture_org_cost_context(user_id, org_id)
        async with trial_cost_context(user_id, context):
            return await function(*args, **kwargs)

    return wrapped
