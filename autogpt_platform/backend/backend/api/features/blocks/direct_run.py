"""Running one block for its caller, as the block-execute routes do.

``POST /api/blocks/{block_id}/execute`` and v1's ``POST /blocks/{block_id}/execute``
both run their block through ``run_block_directly``. A block's ``run`` takes its
credentials as keyword arguments, which ``run_block_directly`` loads from the
caller's own store for the credentials inputs the request names, the way a graph
run's node gets them (``backend.executor.block_credentials``). It checks them
before it charges for the run, so a request refused for its credentials costs
nothing.
"""

import logging
import time
from typing import Literal

from fastapi import HTTPException
from pydantic import ValidationError
from pydantic_core import PydanticUndefined
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_402_PAYMENT_REQUIRED,
    HTTP_503_SERVICE_UNAVAILABLE,
    HTTP_504_GATEWAY_TIMEOUT,
)

from backend.api.utils.validation_errors import bounded_validation_errors
from backend.blocks._base import AnyBlockSchema
from backend.blocks._platform_credentials import (
    is_platform_credential,
    platform_credentials_refused_message,
)
from backend.data.block import BlockInput, CompletedBlockOutput
from backend.data.execution import ExecutionContext
from backend.data.model import CredentialsFieldInfo
from backend.executor import utils as execution_utils
from backend.executor.auto_credentials import (
    MissingAutoCredentialsError,
    acquire_auto_credentials,
)
from backend.executor.block_credentials import (
    check_credentials_input,
    execute_with_credentials,
    names_credentials,
)
from backend.executor.direct_block_execution import (
    DirectBlockExecutionSlot,
    DirectBlockExecutionTimeoutError,
    run_block_off_loop,
)
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.oauth.refresh_failure import CredentialsNeedReconnectError
from backend.integrations.providers import ProviderName
from backend.monitoring.instrumentation import record_block_execution
from backend.util.exceptions import InsufficientBalanceError
from backend.util.logging import TruncatedLogger
from backend.util.request import HTTPClientError

logger = logging.getLogger(__name__)


async def run_block_directly(
    block: AnyBlockSchema,
    input_data: BlockInput,
    *,
    user_id: str,
    execution_context: ExecutionContext,
    source: Literal["internal", "external"],
    slot: DirectBlockExecutionSlot,
) -> CompletedBlockOutput:
    """Run *block* on *input_data* for *user_id*, holding *slot*, and return its
    outputs. ``source`` names the route in the run's credit transaction.

    Raises:
        HTTPException: 400 when a credentials input names none where the block
            needs some, or credentials the caller can't run it with, saying
            what the input takes; 402 when the balance can't pay for the run;
            503 when an OAuth token due a refresh is leased to another run;
            504 when the run outlasts its time budget.
    """
    log_metadata = TruncatedLogger(
        logger,
        prefix=f"[Direct block run {block.name}]",
        metadata={"user_id": user_id, "block_id": block.id},
    )
    await _check_credentials(
        block,
        input_data,
        user_id=user_id,
        user_timezone=execution_context.user_timezone,
        log_metadata=log_metadata,
    )
    try:
        await execution_utils.charge_for_direct_block_execution(
            user_id=user_id, block=block, input_data=input_data, source=source
        )
    except InsufficientBalanceError as e:
        raise HTTPException(status_code=HTTP_402_PAYMENT_REQUIRED, detail=str(e)) from e

    status = "error"
    start_time = time.time()
    try:
        # The credentials are loaded on the worker loop that runs the block,
        # since a lease belongs to the loop that took it.
        output = await run_block_off_loop(
            execute_with_credentials(
                block,
                input_data,
                user_id=user_id,
                execution_context=execution_context,
                log_metadata=log_metadata,
            ),
            slot,
        )
        status = "success"
        return output
    except DirectBlockExecutionTimeoutError as e:
        raise HTTPException(status_code=HTTP_504_GATEWAY_TIMEOUT, detail=str(e)) from e
    finally:
        record_block_execution(
            block_type=block.__class__.__name__,
            status=status,
            duration=time.time() - start_time,
        )


async def _check_credentials(
    block: AnyBlockSchema,
    input_data: BlockInput,
    *,
    user_id: str,
    user_timezone: str,
    log_metadata: TruncatedLogger,
) -> None:
    """Refuse a run with a credentials input the block can't run on, naming
    what that input takes."""
    creds_manager = IntegrationCredentialsManager()
    fields_info = block.input_schema.get_credentials_fields_info()
    for field_name in block.input_schema.get_credentials_fields():
        field_info = fields_info[field_name]
        try:
            problem = await _credentials_problem(
                block,
                field_name,
                field_info,
                input_data,
                user_id=user_id,
                creds_manager=creds_manager,
                log_metadata=log_metadata,
                user_timezone=user_timezone,
            )
        except TimeoutError as e:
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"The credentials in the '{field_name}' input are in use by "
                    "another run while they refresh. Try again shortly."
                ),
            ) from e
        if problem is not None:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=(
                    f"{block.name} needs credentials with "
                    f"{_credentials_needed(block, field_info, input_data)} in its "
                    f"'{field_name}' input. {problem}"
                ),
            )
    await _check_picked_files(
        block, input_data, user_id=user_id, creds_manager=creds_manager
    )


async def _credentials_problem(
    block: AnyBlockSchema,
    field_name: str,
    field_info: CredentialsFieldInfo,
    input_data: BlockInput,
    *,
    user_id: str,
    creds_manager: IntegrationCredentialsManager,
    log_metadata: TruncatedLogger,
    user_timezone: str,
) -> str | None:
    """What stops the block running on the credentials *field_name* names, if
    anything.

    A required input that names none is refused, as graph validation refuses it
    before a graph runs, and so is a platform key the block won't use. The
    credentials an input names are loaded as the run will load them, which
    makes the same ownership and other checks, without leasing them.
    """
    field_value = input_data.get(field_name)
    if not names_credentials(field_value):
        required = field_name in block.input_schema.get_required_fields()
        if required and not field_info.credential_reference_only:
            return "None were given."
        return None
    if not isinstance(field_value, dict):
        return "The input takes an object with the credentials' provider, id and type."
    if is_platform_credential(
        str(field_value["id"])
    ) and block.refuses_platform_credentials_for(input_data):
        return platform_credentials_refused_message(str(field_value.get("provider")))
    try:
        await check_credentials_input(
            block.input_schema,
            field_name,
            input_data,
            user_id=user_id,
            creds_manager=creds_manager,
            log_metadata=log_metadata,
            user_timezone=user_timezone,
        )
    except ValidationError as e:
        errors = bounded_validation_errors(e)
        return "They are invalid: " + "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in errors
        )
    except CredentialsNeedReconnectError as e:
        return str(e)
    except HTTPClientError as e:
        return (
            f"The provider refused to refresh them (HTTP {e.status_code}); "
            "reconnect the account to continue."
        )
    except ValueError as e:
        return str(e)
    return None


async def _check_picked_files(
    block: AnyBlockSchema,
    input_data: BlockInput,
    *,
    user_id: str,
    creds_manager: IntegrationCredentialsManager,
) -> None:
    """Refuse a run with a file input (a picker) that carries no account of
    the caller's, as the run will load that account's credentials."""
    try:
        await acquire_auto_credentials(
            input_model=block.input_schema,
            input_data=input_data,
            creds_manager=creds_manager,
            user_id=user_id,
            lock=False,
        )
    except MissingAutoCredentialsError as e:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=(
                f"{block.name} needs a file picked with one of your connected "
                "accounts: its file input must carry that account's credentials "
                "id as `_credentials_id`."
            ),
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e)) from e


def _credentials_needed(
    block: AnyBlockSchema, field_info: CredentialsFieldInfo, input_data: BlockInput
) -> str:
    """The providers and types a credentials input takes, for this input: a
    field whose provider follows another input (an LLM's model) names the one
    that input picks."""
    if field_info.discriminator:
        choice = input_data.get(field_info.discriminator)
        field = block.input_schema.model_fields.get(field_info.discriminator)
        if choice is None and field and field.default is not PydanticUndefined:
            choice = field.default
        try:
            field_info = field_info.discriminate(choice)
        except (ValueError, TypeError):  # TypeError: an unhashable choice
            pass  # Not a choice the field maps; name every provider it takes.
    providers = sorted(ProviderName(provider).value for provider in field_info.provider)
    types = sorted(field_info.supported_types)
    return (
        f"provider {' or '.join(map(repr, providers))} "
        f"and type {' or '.join(map(repr, types))}"
    )
