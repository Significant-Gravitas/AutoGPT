"""Immutable usage attribution, inherited by child tasks and queued work."""

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import signature
from typing import ParamSpec, TypeVar

from pydantic import BaseModel, ConfigDict, TypeAdapter

from backend.copilot.usage_activation import get_ready_usage_state
from backend.data import db_accessors


class TrialCostContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str | None
    trial_id: str | None
    generation: str | None = None


_trial_cost_context: ContextVar[TrialCostContext | None] = ContextVar(
    "trial_cost_context", default=None
)


async def capture_cost_context(user_id: str | None) -> TrialCostContext:
    existing = get_trial_cost_context(user_id)
    if existing is not None:
        return existing
    if user_id is None:
        return TrialCostContext(user_id=None, trial_id=None)
    state = await get_ready_usage_state(user_id)
    return TrialCostContext(
        user_id=user_id, trial_id=state.trial_id, generation=state.generation
    )


@asynccontextmanager
async def trial_cost_context(
    user_id: str | None, snapshot: TrialCostContext | None = None
) -> AsyncIterator[None]:
    context = snapshot or await capture_cost_context(user_id)
    with restore_cost_context(user_id, context):
        yield


@contextmanager
def restore_cost_context(
    user_id: str | None, context: TrialCostContext
) -> Iterator[None]:
    if context.user_id != user_id:
        raise ValueError("Trial cost attribution belongs to a different user")
    token = _trial_cost_context.set(context)
    try:
        yield
    finally:
        _trial_cost_context.reset(token)


def get_trial_cost_context(user_id: str | None) -> TrialCostContext | None:
    context = _trial_cost_context.get()
    if context is not None and context.user_id != user_id:
        raise ValueError("Trial cost attribution belongs to a different user")
    return context


async def record_attributed_trial_cost(user_id: str, cost_microdollars: int) -> bool:
    context = get_trial_cost_context(user_id)
    if context is None:
        return False
    if context.trial_id is not None:
        await db_accessors.credit_db().record_subscription_trial_cost(
            user_id, cost_microdollars, trial_id=context.trial_id
        )
    return True


P = ParamSpec("P")
R = TypeVar("R")


def attributed_usage(function: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """Capture standalone/background work before its first provider request."""
    parameters = signature(function)

    @wraps(function)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        arguments = parameters.bind(*args, **kwargs)
        arguments.apply_defaults()
        user_id = TypeAdapter(str | None).validate_python(
            arguments.arguments["user_id"]
        )
        async with trial_cost_context(user_id):
            return await function(*args, **kwargs)

    return wrapped
