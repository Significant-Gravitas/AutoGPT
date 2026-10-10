"""
V2 External API - Library route helpers

Shared logic for the endpoints that start runs.
"""

from typing import Any, Mapping

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from starlette import status

from backend.data.credit import get_credit_model
from backend.data.graph import missing_inputs, unknown_inputs

from ..tenancy import TenantContext


async def assert_can_pay(auth: TenantContext) -> None:
    """Refuse a run on a zero balance, as the internal run route does.

    Not the same guard as the paywall inside `add_graph_execution`, which asks
    whether the user has a subscription at all.
    """
    credit_model = await get_credit_model(auth.user_id, auth.organization_id)
    if await credit_model.get_credits(auth.user_id) <= 0:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Insufficient balance to execute the agent. "
            "Please top up your account.",
        )


def assert_inputs_match(
    input_schema: dict[str, Any], inputs: Mapping[str, Any]
) -> None:
    """422 on an input the graph does not have, or a required one omitted or null:
    the executor runs either and finishes with no outputs and no error."""
    errors = [
        {
            "type": "extra_forbidden",
            "loc": ["body", "inputs", name],
            "msg": "Unknown input",
        }
        for name in unknown_inputs(input_schema, inputs)
    ] + [
        {"type": "missing", "loc": ["body", "inputs", name], "msg": "Input required"}
        for name in missing_inputs(input_schema, inputs)
    ]
    if errors:
        raise RequestValidationError(errors)
