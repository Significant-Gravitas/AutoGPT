import asyncio
import logging
import math
import threading
import time
from collections import defaultdict
from concurrent.futures import Future
from typing import Literal, Mapping, Optional, cast

from pydantic import BaseModel, JsonValue, ValidationError

from backend.blocks import get_block
from backend.blocks._base import Block, BlockCostType, BlockType
from backend.copilot.rate_limit import UserPaywalledError, is_user_paywalled
from backend.data import execution as execution_db
from backend.data import graph as graph_db
from backend.data import human_review as human_review_db
from backend.data import onboarding as onboarding_db
from backend.data import user as user_db
from backend.data import workspace as workspace_db

# Import dynamic field utilities from centralized location
from backend.data.block import BlockInput, BlockOutputEntry
from backend.data.block_cost_config import BLOCK_COSTS, compute_token_credits
from backend.data.block_preflight_estimates import get_preflight_estimate
from backend.data.credit import UsageTransactionMetadata, get_user_credit_model
from backend.data.db import prisma
from backend.data.dynamic_fields import merge_execution_input
from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    GraphExecutionMeta,
    GraphExecutionStats,
    GraphExecutionWithNodes,
    NodesInputMasks,
)
from backend.data.graph import GraphModel, Node
from backend.data.model import (
    USER_TIMEZONE_NOT_SET,
    CredentialsMetaInput,
    GraphInput,
    NodeExecutionStats,
)
from backend.data.rabbitmq import Exchange, ExchangeType, Queue, RabbitMQConfig
from backend.util.clients import (
    get_async_execution_event_bus,
    get_async_execution_queue,
    get_database_manager_async_client,
    get_integration_credentials_store,
)
from backend.util.exceptions import (
    GraphNotFoundError,
    GraphValidationError,
    NotFoundError,
)
from backend.util.logging import TruncatedLogger, is_structured_logging_enabled
from backend.util.settings import Config
from backend.util.type import coerce_inputs_to_schema

config = Config()
logger = TruncatedLogger(logging.getLogger(__name__), prefix="[GraphExecutorUtil]")

# ============ Resource Helpers ============ #


class LogMetadata(TruncatedLogger):
    def __init__(
        self,
        logger: logging.Logger,
        user_id: str,
        graph_eid: str,
        graph_id: str,
        node_eid: str,
        node_id: str,
        block_name: str,
        max_length: int = 1000,
    ):
        metadata = {
            "component": "ExecutionManager",
            "user_id": user_id,
            "graph_eid": graph_eid,
            "graph_id": graph_id,
            "node_eid": node_eid,
            "node_id": node_id,
            "block_name": block_name,
        }
        prefix = (
            "[ExecutionManager]"
            if is_structured_logging_enabled()
            else f"[ExecutionManager|uid:{user_id}|gid:{graph_id}|nid:{node_id}]|geid:{graph_eid}|neid:{node_eid}|{block_name}]"  # noqa
        )
        super().__init__(
            logger,
            max_length=max_length,
            prefix=prefix,
            metadata=metadata,
        )


# ============ Execution Cost Helpers ============ #


def execution_usage_cost(execution_count: int) -> tuple[int, int]:
    """
    Calculate the cost of executing a graph based on the current number of node executions.

    Args:
        execution_count: Number of node executions

    Returns:
        Tuple of cost amount and the number of execution count that is included in the cost.
    """
    return (
        (
            config.execution_cost_per_threshold
            if execution_count % config.execution_cost_count_threshold == 0
            else 0
        ),
        config.execution_cost_count_threshold,
    )


def block_usage_cost(
    block: Block,
    input_data: BlockInput,
    data_size: float = 0,
    run_time: float = 0,
    stats: NodeExecutionStats | None = None,
    use_preflight_estimate: bool = True,
) -> tuple[int, BlockInput]:
    """Calculate the credit charge for a block invocation.

    Two calling contexts:
      - Pre-flight (no stats): charge the fixed floor / historical-average
        estimate. SECOND/ITEMS/COST_USD types fall back to 0 only when no
        estimate is registered for the block in `block_preflight_estimates.json`.
      - Post-flight (stats populated): dynamic types consume the captured
        stats to compute the actual charge.

    For SECOND/ITEMS/TOKENS cost entries, ``cost_amount`` is interpreted as
    "credits per ``cost_divisor`` units" — e.g. ``cost_amount=1,
    cost_divisor=10`` under SECOND means "1 credit per 10 seconds".

    ``use_preflight_estimate`` (default True) enables the historical-average
    pre-flight charge for SECOND/ITEMS/COST_USD types. Callers that have NO
    post-flight reconciliation step (e.g. the direct block-execute API
    endpoints, which bypass the executor manager) MUST pass False — otherwise
    the estimate becomes the final charge and users are over- or undercharged
    relative to their actual run cost. The executor pre-flight path keeps the
    default since reconciliation always follows there.
    """
    block_costs = BLOCK_COSTS.get(type(block))
    if not block_costs:
        return 0, {}

    is_preflight = stats is None and run_time == 0 and use_preflight_estimate

    for block_cost in block_costs:
        if not _is_cost_filter_match(block_cost.cost_filter, input_data):
            continue

        if block_cost.cost_type == BlockCostType.RUN:
            return block_cost.cost_amount, block_cost.cost_filter

        if block_cost.cost_type == BlockCostType.BYTE:
            return (
                int(data_size * block_cost.cost_amount),
                block_cost.cost_filter,
            )

        if block_cost.cost_type == BlockCostType.SECOND:
            if is_preflight:
                return get_preflight_estimate(block.id), block_cost.cost_filter
            # Ceil so partial divisor-units still bill — avoids 0-credit leaks
            # on sub-divisor runs (e.g. 1s on a `1cr / 3s` block).
            seconds = _coerce_seconds(run_time, stats)
            credits = (
                math.ceil(seconds / block_cost.cost_divisor) * block_cost.cost_amount
                if seconds > 0
                else 0
            )
            return credits, block_cost.cost_filter

        if block_cost.cost_type == BlockCostType.ITEMS:
            if is_preflight:
                return get_preflight_estimate(block.id), block_cost.cost_filter
            # Ceil so partial buckets still bill — avoids 0-credit leaks on
            # single-item returns under a >1 divisor (e.g. Apollo 1cr/2-items).
            items = _coerce_items(stats)
            credits = (
                math.ceil(items / block_cost.cost_divisor) * block_cost.cost_amount
                if items > 0
                else 0
            )
            return credits, block_cost.cost_filter

        if block_cost.cost_type == BlockCostType.COST_USD:
            if is_preflight:
                return get_preflight_estimate(block.id), block_cost.cost_filter
            usd = _coerce_usd(stats)
            return (
                max(0, math.ceil(usd * block_cost.cost_amount)),
                block_cost.cost_filter,
            )

        if block_cost.cost_type == BlockCostType.TOKENS:
            return (
                compute_token_credits(input_data, stats),
                block_cost.cost_filter,
            )

    return 0, {}


async def charge_for_direct_block_execution(
    user_id: str,
    block: Block,
    input_data: BlockInput,
    *,
    source: Literal["internal", "external"],
) -> None:
    """Pre-flight charge for a direct block-execute API call.

    Shared by both ``POST /api/blocks/{id}/execute`` (internal UI) and
    ``POST /api/v1/blocks/{id}/execute`` (external API key) so the two
    routes stay in lock-step on cost calculation, transaction metadata,
    and 402 mapping. ``source`` is recorded in the credit-history
    ``reason`` so transactions remain attributable to the originating
    surface.

    Dynamic-cost blocks (TOKENS / COST_USD / SECOND / ITEMS) are NOT charged
    on this code path — they return 0 from ``block_usage_cost`` because we
    pass ``use_preflight_estimate=False``. The estimate path is only safe
    when post-flight reconciliation follows (executor/manager.py); the
    direct block-execute API endpoints bypass the manager and have no
    reconciliation step, so charging the estimate would lock in an
    incorrect amount with no chance to settle the delta.
    """
    cost, cost_filter = block_usage_cost(
        block, input_data, use_preflight_estimate=False
    )
    if cost <= 0:
        return
    credit_model = await get_user_credit_model(user_id)
    await credit_model.spend_credits(
        user_id=user_id,
        cost=cost,
        metadata=UsageTransactionMetadata(
            block_id=block.id,
            block=block.name,
            input=cost_filter,
            reason=f"Direct {source} block execution of {block.name}",
        ),
    )


def _coerce_seconds(run_time: float, stats: NodeExecutionStats | None) -> float:
    if run_time > 0:
        return run_time
    if stats and stats.walltime > 0:
        return stats.walltime
    return 0.0


def _coerce_items(stats: NodeExecutionStats | None) -> int:
    if not stats or stats.provider_cost is None:
        return 0
    # provider_cost is a raw item count only when explicitly typed 'items';
    # a None type likely means USD (resolve_tracking defaults), so reject it
    # here to avoid misreading a fractional dollar amount as an item count.
    if stats.provider_cost_type != "items":
        return 0
    return max(0, int(stats.provider_cost))


def _coerce_usd(stats: NodeExecutionStats | None) -> float:
    if not stats or stats.provider_cost is None:
        return 0.0
    # provider_cost is billable only when tagged as cost_usd — otherwise it
    # encodes a non-dollar quantity (e.g. items) that would wildly over-bill.
    if stats.provider_cost_type and stats.provider_cost_type != "cost_usd":
        return 0.0
    return max(0.0, float(stats.provider_cost))


def _is_cost_filter_match(cost_filter: BlockInput, input_data: BlockInput) -> bool:
    """
    Filter rules:
      - If cost_filter is an object, then check if cost_filter is the subset of input_data
      - Otherwise, check if cost_filter is equal to input_data.
      - Undefined, null, and empty string are considered as equal.
    """
    if not isinstance(cost_filter, dict) or not isinstance(input_data, dict):
        return cost_filter == input_data

    return all(
        (not input_data.get(k) and not v)
        or (input_data.get(k) and _is_cost_filter_match(v, input_data[k]))
        for k, v in cost_filter.items()
    )


# ============ Execution Input Helpers ============ #

# Dynamic field utilities are now imported from backend.data.dynamic_fields


def validate_exec(
    node: Node,
    data: BlockInput,
    resolve_input: bool = True,
    dry_run: bool = False,
) -> tuple[BlockInput | None, str]:
    """
    Validate the input data for a node execution.

    Args:
        node: The node to execute.
        data: The input data for the node execution.
        resolve_input: Whether to resolve dynamic pins into dict/list/object.
        dry_run: When True, credential fields are allowed to be missing — they
            will be substituted with a sentinel so the node can be queued and
            later executed via simulate_block.

    Returns:
        A tuple of the validated data and the block name.
        If the data is invalid, the first element will be None, and the second element
        will be an error message.
        If the data is valid, the first element will be the resolved input data, and
        the second element will be the block name.
    """
    node_block = get_block(node.block_id)
    if not node_block:
        return None, f"Block for {node.block_id} not found."
    schema = node_block.input_schema

    # Input data (without default values) should contain all required fields.
    error_prefix = f"Input data missing or mismatch for `{node_block.name}`:"
    if missing_links := schema.get_missing_links(data, node.input_links):
        return None, f"{error_prefix} unpopulated links {missing_links}"

    # For dry runs, supply sentinel values for any missing credential fields so
    # the node can be queued — simulate_block never calls the real API anyway.
    if dry_run:
        cred_field_names = set(schema.get_credentials_fields().keys())
        for field_name in cred_field_names:
            if field_name not in data:
                data = {**data, field_name: None}

    # Merge input data with default values and resolve dynamic dict/list/object pins.
    input_default = schema.get_input_defaults(node.input_default)
    data = {**input_default, **data}
    if resolve_input:
        data = merge_execution_input(data)

    # Coerce non-matching data types to the expected input schema.
    coerce_inputs_to_schema(data, schema)

    # Input data post-merge should contain all required fields from the schema.
    if missing_input := schema.get_missing_input(data):
        if dry_run:
            # In dry-run mode all missing inputs are tolerated — simulate_block()
            # generates synthetic outputs without needing real input values.
            pass
        else:
            return None, f"{error_prefix} missing input {missing_input}"

    # Last validation: Validate the input values against the schema.
    # Skip for dry runs — simulate_block doesn't use real inputs, and sentinel
    # credential values (None) would fail JSON-schema type/required checks.
    if not dry_run:
        if error := schema.get_mismatch_error(data):
            error_message = f"{error_prefix} {error}"
            logger.warning(error_message)
            return None, error_message

    return data, node_block.name


# ---------------------------------------------------------------------------
# Credential validation error message templates.
#
# These constants are the single source of truth for the error messages
# emitted by ``_validate_node_input_credentials``.  Both the raise sites
# below and the public matcher ``is_credential_validation_error_message``
# reference them, so adding a new credential error means adding a
# constant here — the matcher and tests stay in sync automatically.
#
# If you add a new credential error string, also add its constant to
# ``_CREDENTIAL_ERROR_MARKERS`` below so the copilot's credential-race
# fallback continues to recognise it.
# ---------------------------------------------------------------------------
CRED_ERR_REQUIRED = "These credentials are required"
CRED_ERR_INVALID_PREFIX = "Invalid credentials:"
CRED_ERR_INVALID_TYPE_MISMATCH = "Invalid credentials: type/provider mismatch"
CRED_ERR_NOT_AVAILABLE_PREFIX = "Credentials not available:"
CRED_ERR_UNKNOWN_PREFIX = "Unknown credentials #"

# Markers used by ``is_credential_validation_error_message`` to classify a
# message. Each entry is (match_mode, lowercased_marker) — "exact" means
# the full message must equal the marker, "prefix" means it must start
# with the marker.
_MatchMode = Literal["exact", "prefix"]
_CREDENTIAL_ERROR_MARKERS: tuple[tuple[_MatchMode, str], ...] = (
    ("exact", CRED_ERR_REQUIRED.lower()),
    # NOTE: CRED_ERR_INVALID_TYPE_MISMATCH is intentionally omitted here —
    # the "prefix" entry for CRED_ERR_INVALID_PREFIX already covers it (since
    # CRED_ERR_INVALID_TYPE_MISMATCH starts with "Invalid credentials:").
    ("prefix", CRED_ERR_INVALID_PREFIX.lower()),
    ("prefix", CRED_ERR_NOT_AVAILABLE_PREFIX.lower()),
    ("prefix", CRED_ERR_UNKNOWN_PREFIX.lower()),
)


def is_credential_validation_error_message(message: str) -> bool:
    """Return True if *message* came from the credential gate in
    :func:`_validate_node_input_credentials`.

    Kept as a public module-level helper so other layers (e.g. the
    copilot tool that rebuilds the inline credentials setup card on a
    credential race) can distinguish credential failures from other
    graph validation errors without redefining the string list.

    Drift prevention: raise sites and this matcher both reference the
    ``CRED_ERR_*`` constants defined above, and
    ``test_credential_error_markers_cover_all_raise_sites`` exercises
    every branch of ``_validate_node_input_credentials`` to assert the
    emitted messages are recognised.
    """
    lower = message.lower()
    for mode, marker in _CREDENTIAL_ERROR_MARKERS:
        if mode == "exact" and lower == marker:
            return True
        if mode == "prefix" and lower.startswith(marker):
            return True
    return False


async def _validate_node_input_credentials(
    graph: GraphModel,
    user_id: str,
    nodes_input_masks: Optional[NodesInputMasks] = None,
) -> tuple[dict[str, dict[str, str]], set[str]]:
    """
    Checks all credentials for all nodes of the graph and returns structured errors
    and a set of nodes that should be skipped due to optional missing credentials.

    Returns:
        tuple[
            dict[node_id, dict[field_name, error_message]]: Credential validation errors per node,
            set[node_id]: Nodes that should be skipped (optional credentials not configured)
        ]
    """
    credential_errors: dict[str, dict[str, str]] = defaultdict(dict)
    nodes_to_skip: set[str] = set()

    for node in graph.nodes:
        block = node.block

        # Find any fields of type CredentialsMetaInput
        credentials_fields = block.input_schema.get_credentials_fields()
        auto_credentials_fields = block.input_schema.get_auto_credentials_fields()
        if not credentials_fields and not auto_credentials_fields:
            continue

        # Track if any credential field is missing for this node
        has_missing_credentials = False

        # Local helper: mark the node as skippable when a per-field branch
        # decides the field is optional-and-missing. We add to
        # `nodes_to_skip` here rather than relying on the post-loop
        # guard — that guard only fires when the NODE-level
        # ``is_creds_optional`` is True. For auto-credential fields the
        # optionality is usually field-level (``field_name not in
        # required_fields`` because the schema default is None), so
        # deferring would let the node silently pass validation and then
        # crash in ``_acquire_auto_credentials`` at runtime. See Cursor
        # thread PRRT_kwDOJKSTjM58r_37. Defined once per node (not per
        # field) to avoid redefining the closure each inner-loop
        # iteration — see Cursor thread PRRT_kwDOJKSTjM58sEDe.
        def _mark_optional_skip() -> None:
            nonlocal has_missing_credentials
            has_missing_credentials = True
            nodes_to_skip.add(node.id)

        # A credential field is optional if the node metadata says so, or if
        # the block schema declares a default for the field.
        required_fields = block.input_schema.get_required_fields()
        is_creds_optional = node.credentials_optional

        for field_name, credentials_meta_type in credentials_fields.items():
            field_is_optional = is_creds_optional or field_name not in required_fields
            try:
                # Check nodes_input_masks first, then input_default
                field_value = None
                if (
                    nodes_input_masks
                    and (node_input_mask := nodes_input_masks.get(node.id))
                    and field_name in node_input_mask
                ):
                    field_value = node_input_mask[field_name]
                elif field_name in node.input_default:
                    # For optional credentials, don't use input_default - treat as missing
                    # This prevents stale credential IDs from failing validation
                    if field_is_optional:
                        field_value = None
                    else:
                        field_value = node.input_default[field_name]

                # Check if credentials are missing (None, empty, or not present)
                if field_value is None or (
                    isinstance(field_value, dict) and not field_value.get("id")
                ):
                    has_missing_credentials = True
                    # If credential field is optional, skip instead of error
                    if field_is_optional:
                        continue  # Don't add error, will be marked for skip after loop
                    else:
                        credential_errors[node.id][field_name] = CRED_ERR_REQUIRED
                        continue

                credentials_meta = credentials_meta_type.model_validate(field_value)

            except ValidationError as e:
                # Validation error means credentials were provided but invalid
                # This should always be an error, even if optional
                credential_errors[node.id][
                    field_name
                ] = f"{CRED_ERR_INVALID_PREFIX} {e}"
                continue

            try:
                # Fetch the corresponding Credentials and perform sanity checks
                credentials = await get_integration_credentials_store().get_creds_by_id(
                    user_id, credentials_meta.id
                )
            except Exception as e:
                # Handle any errors fetching credentials
                # If credentials were explicitly configured but unavailable, it's an error
                credential_errors[node.id][
                    field_name
                ] = f"{CRED_ERR_NOT_AVAILABLE_PREFIX} {e}"
                continue

            if not credentials:
                credential_errors[node.id][
                    field_name
                ] = f"{CRED_ERR_UNKNOWN_PREFIX}{credentials_meta.id}"
                continue

            if (
                credentials.provider != credentials_meta.provider
                or credentials.type != credentials_meta.type
            ):
                logger.warning(
                    f"Invalid credentials #{credentials.id} for node #{node.id}: "
                    "type/provider mismatch: "
                    f"{credentials_meta.type}<>{credentials.type};"
                    f"{credentials_meta.provider}<>{credentials.provider}"
                )
                credential_errors[node.id][field_name] = CRED_ERR_INVALID_TYPE_MISMATCH
                continue

        # Validate auto-credentials (GoogleDriveFileField-based)
        # These have _credentials_id embedded in the file field data
        if auto_credentials_fields:
            for _kwarg_name, info in auto_credentials_fields.items():
                field_name = info["field_name"]
                field_is_optional = (
                    is_creds_optional or field_name not in required_fields
                )
                # Check input_default and nodes_input_masks for the field value
                field_value = node.input_default.get(field_name)
                if nodes_input_masks and node.id in nodes_input_masks:
                    field_value = nodes_input_masks[node.id].get(
                        field_name, field_value
                    )

                if field_value is None:
                    # Sentry HIGH: an explicitly-None value (e.g. cleared by
                    # `_reassign_ids` on fork, or nulled by a mask) means
                    # credentials were there and are now gone. Treat as
                    # missing so optional fields hit `nodes_to_skip` and
                    # required fields surface a clean re-auth message —
                    # don't silently fall through to `_acquire_auto_credentials`
                    # which would then crash with ValueError at runtime.
                    # NOTE: this branch only fires when the key is
                    # explicitly `None`. If the field is absent from
                    # `input_default` altogether (chained from upstream
                    # via `input_links`), `.get()` also returns None — but
                    # that path is handled at execute time by
                    # `_acquire_auto_credentials` skipping fields not in
                    # `input_data`. To keep this validator from over-reaching
                    # in that case, callers set the field explicitly to
                    # `None` only for the cleared-fork scenario.
                    field_is_explicitly_none = field_name in node.input_default or (
                        nodes_input_masks
                        and node.id in nodes_input_masks
                        and field_name in nodes_input_masks[node.id]
                    )
                    if not field_is_explicitly_none:
                        continue
                    if field_is_optional:
                        _mark_optional_skip()
                        continue
                    has_missing_credentials = True
                    credential_errors[node.id][field_name] = (
                        f"{CRED_ERR_NOT_AVAILABLE_PREFIX} no file selected "
                        "for this field. Please select a file via the "
                        "picker to authenticate."
                    )
                    continue

                if field_value and isinstance(field_value, dict):
                    if "_credentials_id" not in field_value:
                        # Key removed (e.g., on fork) — needs re-auth. Use the
                        # CRED_ERR_NOT_AVAILABLE_PREFIX marker so the copilot
                        # credential-race fallback recognises this as a
                        # credentials gate failure.
                        if field_is_optional:
                            _mark_optional_skip()
                            continue
                        has_missing_credentials = True
                        credential_errors[node.id][field_name] = (
                            f"{CRED_ERR_NOT_AVAILABLE_PREFIX} authentication "
                            "missing for the selected file. Please re-select "
                            "the file to authenticate with your own account."
                        )
                        continue
                    cred_id = field_value.get("_credentials_id")
                    if cred_id is None:
                        # Explicitly None means the value is being chained in
                        # at execution time from an upstream block — skip.
                        continue
                    if not isinstance(cred_id, str) or not cred_id.strip():
                        # Non-string or empty string is a corrupted state —
                        # treat it like a missing credential so the user
                        # re-authenticates rather than silently running with
                        # no creds.
                        if field_is_optional:
                            _mark_optional_skip()
                            continue
                        has_missing_credentials = True
                        credential_errors[node.id][field_name] = (
                            f"{CRED_ERR_NOT_AVAILABLE_PREFIX} credential id "
                            "on the selected file is empty or invalid. "
                            "Please re-select the file."
                        )
                        continue
                    try:
                        creds_store = get_integration_credentials_store()
                        creds = await creds_store.get_creds_by_id(user_id, cred_id)
                    except Exception as e:
                        if field_is_optional:
                            _mark_optional_skip()
                            continue
                        has_missing_credentials = True
                        credential_errors[node.id][
                            field_name
                        ] = f"{CRED_ERR_NOT_AVAILABLE_PREFIX} {e}"
                        continue
                    if not creds:
                        if field_is_optional:
                            _mark_optional_skip()
                            continue
                        has_missing_credentials = True
                        credential_errors[node.id][
                            field_name
                        ] = f"{CRED_ERR_UNKNOWN_PREFIX}{cred_id}"

        # If node has optional credentials and any are missing, skip the
        # node so the executor doesn't try to execute it with None creds.
        # The per-field loops above deliberately didn't record an error for
        # the optional case — the "will be marked for skip after loop"
        # contract lives here.
        if (
            has_missing_credentials
            and is_creds_optional
            and node.id not in credential_errors
        ):
            logger.info(
                f"Node #{node.id}: optional credentials not configured, skipping"
            )
            nodes_to_skip.add(node.id)

    return credential_errors, nodes_to_skip


def make_node_credentials_input_map(
    graph: GraphModel,
    graph_credentials_input: Mapping[str, CredentialsMetaInput],
) -> NodesInputMasks:
    """
    Maps credentials for an execution to the correct nodes.

    Params:
        graph: The graph to be executed.
        graph_credentials_input: A (graph_input_name, credentials_meta) map.

    Returns:
        dict[node_id, dict[field_name, CredentialsMetaRaw]]: Node credentials input map.
    """
    result: dict[str, dict[str, JsonValue]] = {}

    # Only map regular credentials (not auto_credentials, which are resolved
    # at execution time from _credentials_id in file field data)
    graph_cred_inputs = graph.regular_credentials_inputs

    for graph_input_name, (_, compatible_node_fields, _) in graph_cred_inputs.items():
        # Best-effort map: skip missing items
        if graph_input_name not in graph_credentials_input:
            continue

        # Use passed-in credentials for all compatible node input fields
        for node_id, node_field_name in compatible_node_fields:
            if node_id not in result:
                result[node_id] = {}
            result[node_id][node_field_name] = graph_credentials_input[
                graph_input_name
            ].model_dump(exclude_none=True)

    return result


async def validate_graph_with_credentials(
    graph: GraphModel,
    user_id: str,
    nodes_input_masks: Optional[NodesInputMasks] = None,
) -> tuple[Mapping[str, Mapping[str, str]], set[str]]:
    """
    Validate graph including credentials and return structured errors per node,
    along with a set of nodes that should be skipped due to optional missing credentials.

    Returns:
        tuple[
            dict[node_id, dict[field_name, error_message]]: Validation errors per node,
            set[node_id]: Nodes that should be skipped (optional credentials not configured)
        ]
    """
    # Get input validation errors
    node_input_errors = GraphModel.validate_graph_get_errors(
        graph, for_run=True, nodes_input_masks=nodes_input_masks
    )

    # Get credential input/availability/validation errors and nodes to skip
    (
        node_credential_input_errors,
        nodes_to_skip,
    ) = await _validate_node_input_credentials(graph, user_id, nodes_input_masks)

    # Merge credential errors with structural errors
    for node_id, field_errors in node_credential_input_errors.items():
        if node_id not in node_input_errors:
            node_input_errors[node_id] = {}
        node_input_errors[node_id].update(field_errors)

    return node_input_errors, nodes_to_skip


async def _construct_starting_node_execution_input(
    graph: GraphModel,
    user_id: str,
    graph_inputs: GraphInput,
    nodes_input_masks: Optional[NodesInputMasks] = None,
    dry_run: bool = False,
) -> tuple[list[tuple[str, BlockInput]], set[str]]:
    """
    Validates and prepares the input data for executing a graph.
    This function checks the graph for starting nodes, validates the input data
    against the schema, and resolves dynamic input pins into a single list,
    dictionary, or object.

    Args:
        graph (GraphModel): The graph model to execute.
        user_id (str): The ID of the user executing the graph.
        data (GraphInput): The input data for the graph execution.
        node_credentials_map: `dict[node_id, dict[input_name, CredentialsMetaInput]]`
        dry_run: When True, skip credential validation errors (simulation needs no real creds).

    Returns:
        tuple[
            list[tuple[str, BlockInput]]: A list of tuples, each containing the node ID
                and the corresponding input data for that node.
            set[str]: Node IDs that should be skipped (optional credentials not configured)
        ]
    """
    # Use new validation function that includes credentials
    validation_errors, nodes_to_skip = await validate_graph_with_credentials(
        graph, user_id, nodes_input_masks
    )
    # Dry runs simulate every block — missing credentials are irrelevant.
    # Strip credential-only errors so the graph can proceed.
    if dry_run and validation_errors:
        validation_errors = {
            node_id: {
                field: msg
                for field, msg in errors.items()
                if not is_credential_validation_error_message(msg)
            }
            for node_id, errors in validation_errors.items()
        }
        # Remove nodes that have no remaining errors
        validation_errors = {
            node_id: errors for node_id, errors in validation_errors.items() if errors
        }
    n_error_nodes = len(validation_errors)
    n_errors = sum(len(errors) for errors in validation_errors.values())
    if validation_errors:
        raise GraphValidationError(
            f"Graph validation failed: {n_errors} issues on {n_error_nodes} nodes",
            node_errors=validation_errors,
        )

    nodes_input = []
    for node in graph.starting_nodes:
        input_data = {}
        block = node.block

        # Note block should never be executed.
        if block.block_type == BlockType.NOTE:
            continue

        # Extract request input data, and assign it to the input pin.
        if block.block_type == BlockType.INPUT:
            input_name = cast(str | None, node.input_default.get("name"))
            if input_name and input_name in graph_inputs:
                input_data = {"value": graph_inputs[input_name]}

        # Apply node input overrides
        if nodes_input_masks and (node_input_mask := nodes_input_masks.get(node.id)):
            input_data.update(node_input_mask)

        # Webhook-triggered agents cannot be executed directly without payload data.
        # Legitimate webhook triggers provide payload via nodes_input_masks above.
        if (
            block.block_type
            in (
                BlockType.WEBHOOK,
                BlockType.WEBHOOK_MANUAL,
            )
            and "payload" not in input_data
        ):
            raise ValueError(
                "This agent is triggered by an external event (webhook) "
                "and cannot be executed directly. "
                "Please use the appropriate trigger to run this agent."
            )

        input_data, error = validate_exec(node, input_data, dry_run=dry_run)
        if input_data is None:
            raise ValueError(error)
        else:
            nodes_input.append((node.id, input_data))

    if not nodes_input:
        raise ValueError(
            "No starting nodes found for the graph, make sure an AgentInput or blocks with no inbound links are present as starting nodes."
        )

    return nodes_input, nodes_to_skip


async def validate_and_construct_node_execution_input(
    graph_id: str,
    user_id: str,
    graph_inputs: GraphInput,
    graph_version: Optional[int] = None,
    graph_credentials_inputs: Optional[Mapping[str, CredentialsMetaInput]] = None,
    nodes_input_masks: Optional[NodesInputMasks] = None,
    is_sub_graph: bool = False,
    dry_run: bool = False,
) -> tuple[GraphModel, list[tuple[str, BlockInput]], NodesInputMasks, set[str]]:
    """
    Public wrapper that handles graph fetching, credential mapping, and validation+construction.
    This centralizes the logic used by both scheduler validation and actual execution.

    Args:
        graph_id: The ID of the graph to validate/construct.
        user_id: The ID of the user.
        graph_inputs: The input data for the graph execution.
        graph_version: The version of the graph to use.
        graph_credentials_inputs: Credentials inputs to use.
        nodes_input_masks: Node inputs to use.

    Returns:
        GraphModel: Full graph object for the given `graph_id`.
        list[tuple[node_id, BlockInput]]: Starting node IDs with corresponding inputs.
        dict[str, BlockInput]: Node input masks including all passed-in credentials.
        set[str]: Node IDs that should be skipped (optional credentials not configured).

    Raises:
        NotFoundError: If the graph is not found.
        GraphValidationError: If the graph has validation issues.
        ValueError: If there are other validation errors.
    """
    if prisma.is_connected():
        gdb = graph_db
    else:
        gdb = get_database_manager_async_client()

    graph: GraphModel | None = await gdb.get_graph(
        graph_id=graph_id,
        user_id=user_id,
        version=graph_version,
        include_subgraphs=True,
        # Execution/access permission is checked by validate_graph_execution_permissions
        skip_access_check=True,
    )
    if not graph:
        raise GraphNotFoundError(f"Graph #{graph_id} not found.")

    # Validate that the user has permission to execute this graph
    # This checks both library membership and execution permissions,
    # raising specific exceptions for appropriate error handling.
    await gdb.validate_graph_execution_permissions(
        user_id=user_id,
        graph_id=graph.id,
        graph_version=graph.version,
        is_sub_graph=is_sub_graph,
    )

    nodes_input_masks = _merge_nodes_input_masks(
        (
            make_node_credentials_input_map(graph, graph_credentials_inputs)
            if graph_credentials_inputs
            else {}
        ),
        nodes_input_masks or {},
    )

    (
        starting_nodes_input,
        nodes_to_skip,
    ) = await _construct_starting_node_execution_input(
        graph=graph,
        user_id=user_id,
        graph_inputs=graph_inputs,
        nodes_input_masks=nodes_input_masks,
        dry_run=dry_run,
    )

    return graph, starting_nodes_input, nodes_input_masks, nodes_to_skip


def _merge_nodes_input_masks(
    overrides_map_1: NodesInputMasks,
    overrides_map_2: NodesInputMasks,
) -> NodesInputMasks:
    """Perform a per-node merge of input overrides"""
    result = dict(overrides_map_1).copy()
    for node_id, overrides2 in overrides_map_2.items():
        if node_id in result:
            result[node_id] = {**result[node_id], **overrides2}
        else:
            result[node_id] = overrides2
    return result


# ============ Execution Queue Helpers ============ #

GRAPH_EXECUTION_EXCHANGE = Exchange(
    name="graph_execution",
    type=ExchangeType.DIRECT,
    durable=True,
    auto_delete=False,
)
# ``_v2`` suffix marks the classic→quorum rollover; old-image consumers
# drain the unsuffixed queue. Orphans cleaned up in a follow-up PR.
GRAPH_EXECUTION_QUEUE_NAME = "graph_execution_queue_v2"
GRAPH_EXECUTION_ROUTING_KEY = "graph_execution.run"

GRAPH_EXECUTION_CANCEL_EXCHANGE = Exchange(
    name="graph_execution_cancel",
    type=ExchangeType.FANOUT,
    durable=True,
    auto_delete=True,
)
GRAPH_EXECUTION_CANCEL_QUEUE_NAME = "graph_execution_cancel_queue_v2"

# Graceful shutdown timeout constants
# Agent executions can run for up to 1 day, so we need a graceful shutdown period
# that allows long-running executions to complete naturally
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 24 * 60 * 60  # 1 day to complete active executions


def create_execution_queue_config() -> RabbitMQConfig:
    """
    Define two exchanges and queues:
    - 'graph_execution' (DIRECT) for run tasks.
    - 'graph_execution_cancel' (FANOUT) for cancel requests.
    """
    run_queue = Queue(
        name=GRAPH_EXECUTION_QUEUE_NAME,
        exchange=GRAPH_EXECUTION_EXCHANGE,
        routing_key=GRAPH_EXECUTION_ROUTING_KEY,
        durable=True,
        auto_delete=False,
        arguments={
            # Quorum (not classic mirrored) for leader election + stronger
            # replication across RabbitMQ 4.x cluster nodes.
            "x-queue-type": "quorum",
            # x-consumer-timeout (24h)
            # Problem: Default 30-minute consumer timeout kills long-running graph executions
            # Original error: "Consumer acknowledgement timed out after 1800000 ms (30 minutes)"
            # Solution: Disable consumer timeout entirely - let graphs run indefinitely
            # Safety: Heartbeat mechanism now handles dead consumer detection instead
            # Use case: Graph executions that take hours to complete (AI model training, etc.)
            "x-consumer-timeout": GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS * 1000,
        },
    )
    cancel_queue = Queue(
        name=GRAPH_EXECUTION_CANCEL_QUEUE_NAME,
        exchange=GRAPH_EXECUTION_CANCEL_EXCHANGE,
        routing_key="",  # not used for FANOUT
        durable=True,
        auto_delete=False,
        arguments={"x-queue-type": "quorum"},
    )
    return RabbitMQConfig(
        vhost="/",
        exchanges=[GRAPH_EXECUTION_EXCHANGE, GRAPH_EXECUTION_CANCEL_EXCHANGE],
        queues=[run_queue, cancel_queue],
    )


class CancelExecutionEvent(BaseModel):
    graph_exec_id: str


async def _get_child_executions(parent_exec_id: str) -> list["GraphExecutionMeta"]:
    """
    Get all child executions of a parent execution using the execution_db pattern.

    Args:
        parent_exec_id: Parent graph execution ID

    Returns:
        List of child graph executions
    """
    from backend.data.db import prisma

    if prisma.is_connected():
        edb = execution_db
    else:
        edb = get_database_manager_async_client()

    return await edb.get_child_graph_executions(parent_exec_id)


async def stop_graph_execution(
    user_id: str,
    graph_exec_id: str,
    wait_timeout: float = 15.0,
    cascade: bool = True,
):
    """
    Stop a graph execution and optionally all its child executions.

    Mechanism:
    1. Set the cancel event for this execution
    2. If cascade=True, recursively stop all child executions
    3. Graph executor's cancel handler thread detects the event, terminates workers,
       reinitializes worker pool, and returns.
    4. Update execution statuses in DB and set `error` outputs to `"TERMINATED"`.

    Args:
        user_id: User ID who owns the execution
        graph_exec_id: Graph execution ID to stop
        wait_timeout: Maximum time to wait for execution to stop (seconds)
        cascade: If True, recursively stop all child executions
    """
    queue_client = await get_async_execution_queue()
    db = execution_db if prisma.is_connected() else get_database_manager_async_client()

    # First, find and stop all child executions if cascading
    if cascade:
        children = await _get_child_executions(graph_exec_id)
        logger.info(
            f"Stopping {len(children)} child executions of execution {graph_exec_id}"
        )

        # Stop all children in parallel (recursively, with cascading enabled)
        if children:
            await asyncio.gather(
                *[
                    stop_graph_execution(
                        user_id=user_id,
                        graph_exec_id=child.id,
                        wait_timeout=wait_timeout,
                        cascade=True,  # Recursively cascade to grandchildren
                    )
                    for child in children
                ],
                return_exceptions=True,  # Don't fail parent stop if child stop fails
            )

    # Now stop this execution
    await queue_client.publish_message(
        routing_key="",
        message=CancelExecutionEvent(graph_exec_id=graph_exec_id).model_dump_json(),
        exchange=GRAPH_EXECUTION_CANCEL_EXCHANGE,
    )

    if not wait_timeout:
        return

    start_time = time.time()
    while time.time() - start_time < wait_timeout:
        graph_exec = await db.get_graph_execution_meta(
            execution_id=graph_exec_id, user_id=user_id
        )

        if not graph_exec:
            raise NotFoundError(f"Graph execution #{graph_exec_id} not found.")

        if graph_exec.status in [
            ExecutionStatus.TERMINATED,
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
        ]:
            # If graph execution is terminated/completed/failed, cancellation is complete
            await get_async_execution_event_bus().publish(graph_exec)
            return

        if graph_exec.status in [
            ExecutionStatus.QUEUED,
            ExecutionStatus.INCOMPLETE,
            ExecutionStatus.REVIEW,
        ]:
            # If the graph is queued/incomplete/paused for review, terminate immediately
            # No need to wait for executor since it's not actively running

            # If graph is in REVIEW status, clean up pending reviews before terminating
            if graph_exec.status == ExecutionStatus.REVIEW:
                # Use human_review_db if Prisma connected, else database manager
                review_db = (
                    human_review_db
                    if prisma.is_connected()
                    else get_database_manager_async_client()
                )
                # Mark all pending reviews as rejected/cancelled
                cancelled_count = await review_db.cancel_pending_reviews_for_execution(
                    graph_exec_id, user_id
                )
                logger.info(
                    f"Cancelled {cancelled_count} pending review(s) for stopped execution {graph_exec_id}"
                )

            graph_exec.status = ExecutionStatus.TERMINATED

            await asyncio.gather(
                # Update graph execution status
                db.update_graph_execution_stats(
                    graph_exec_id=graph_exec.id,
                    status=ExecutionStatus.TERMINATED,
                ),
                # Publish graph execution event
                get_async_execution_event_bus().publish(graph_exec),
            )
            return

        if graph_exec.status == ExecutionStatus.RUNNING:
            await asyncio.sleep(0.1)

    raise TimeoutError(
        f"Graph execution #{graph_exec_id} will need to take longer than {wait_timeout} seconds to stop. "
        f"You can check the status of the execution in the UI or try again later."
    )


async def add_graph_execution(
    graph_id: str,
    user_id: str,
    inputs: Optional[GraphInput] = None,
    preset_id: Optional[str] = None,
    graph_version: Optional[int] = None,
    graph_credentials_inputs: Optional[Mapping[str, CredentialsMetaInput]] = None,
    nodes_input_masks: Optional[NodesInputMasks] = None,
    execution_context: Optional[ExecutionContext] = None,
    graph_exec_id: Optional[str] = None,
    dry_run: bool = False,
    *,
    bypass_paywall: bool = False,
) -> GraphExecutionWithNodes:
    """
    Adds a graph execution to the queue and returns the execution entry.

    Supports two modes:
    1. CREATE mode (graph_exec_id=None): Validates, creates new DB entry, and queues
    2. REQUEUE mode (graph_exec_id provided): Fetches existing execution and re-queues it

    Args:
        graph_id: The ID of the graph to execute.
        user_id: The ID of the user executing the graph.
        inputs: The input data for the graph execution.
        preset_id: The ID of the preset to use.
        graph_version: The version of the graph to execute.
        graph_credentials_inputs: Credentials inputs to use in the execution.
            Keys should map to the keys generated by `GraphModel.aggregate_credentials_inputs`.
        nodes_input_masks: Node inputs to use in the execution.
        parent_graph_exec_id: The ID of the parent graph execution (for nested executions).
        graph_exec_id: If provided, resume this existing execution instead of creating a new one.
        bypass_paywall: Skip the per-user paywall check. Set ONLY for admin
            recovery paths (requeueing stuck executions on behalf of a user
            who may be on NO_TIER) — never for user-initiated runs.
    Returns:
        GraphExecutionWithNodes: The execution entry.
    Raises:
        ValueError: If the graph is not found or if there are validation errors.
        NotFoundError: If graph_exec_id is provided but execution is not found.
        UserPaywalledError: If the user is on NO_TIER and ``ENABLE_PLATFORM_PAYMENT``
            is on for them, **unless** ``bypass_paywall=True``. Raised here
            so every entry point — HTTP routes, scheduled cron, webhook
            triggers, external API, internal copilot tools — gets the same
            gate without each having to remember a route-level dependency.
        Exception: Tier-lookup errors propagate as-is. The HTTP routes that
            call into ``add_graph_execution`` already wrap with
            ``enforce_payment_paywall`` upstream (which maps lookup failure
            to 503), so by the time we get here those callers have a fresh
            check. Background callers (scheduled jobs, webhook handlers,
            copilot tool runs) catch the exception in their own retry
            framework — failing now is preferable to silently giving a
            paywalled user a free run during an outage.
    """
    if not bypass_paywall and await is_user_paywalled(user_id):
        raise UserPaywalledError("A subscription is required to run agents.")

    if prisma.is_connected():
        edb = execution_db
        udb = user_db
        gdb = graph_db
        odb = onboarding_db
        wdb = workspace_db
    else:
        edb = udb = gdb = odb = wdb = get_database_manager_async_client()

    # Get or create the graph execution
    if graph_exec_id:
        # Resume existing execution
        graph_exec = await edb.get_graph_execution(
            user_id=user_id,
            execution_id=graph_exec_id,
            include_node_executions=True,
        )

        if not graph_exec:
            raise NotFoundError(f"Graph execution #{graph_exec_id} not found.")

        # Use existing execution's compiled input masks
        compiled_nodes_input_masks = graph_exec.nodes_input_masks or {}
        # For resumed executions, nodes_to_skip was already determined at creation time
        # TODO: Consider storing nodes_to_skip in DB if we need to preserve it across resumes
        nodes_to_skip: set[str] = set()

        logger.info(f"Resuming graph execution #{graph_exec.id} for graph #{graph_id}")
    else:
        parent_exec_id = (
            execution_context.parent_execution_id if execution_context else None
        )

        # When execution_context is provided (e.g. from AgentExecutorBlock),
        # inherit dry_run so child-graph validation skips credential checks.
        if execution_context and execution_context.dry_run:
            dry_run = True

        # Create new execution
        (
            graph,
            starting_nodes_input,
            compiled_nodes_input_masks,
            nodes_to_skip,
        ) = await validate_and_construct_node_execution_input(
            graph_id=graph_id,
            user_id=user_id,
            graph_inputs=inputs or {},
            graph_version=graph_version,
            graph_credentials_inputs=graph_credentials_inputs,
            nodes_input_masks=nodes_input_masks,
            is_sub_graph=parent_exec_id is not None,
            dry_run=dry_run,
        )

        graph_exec = await edb.create_graph_execution(
            user_id=user_id,
            graph_id=graph_id,
            graph_version=graph.version,
            inputs=inputs or {},
            credential_inputs=graph_credentials_inputs,
            nodes_input_masks=nodes_input_masks,
            starting_nodes_input=starting_nodes_input,
            preset_id=preset_id,
            parent_graph_exec_id=parent_exec_id,
            is_dry_run=dry_run,
        )

        logger.info(
            f"Created graph execution #{graph_exec.id} for graph "
            f"#{graph_id} with {len(starting_nodes_input)} starting nodes"
        )

    # Generate execution context if it's not provided
    if execution_context is None:
        user = await udb.get_user_by_id(user_id)
        settings = await gdb.get_graph_settings(user_id=user_id, graph_id=graph_id)
        workspace = await wdb.get_or_create_workspace(user_id)

        execution_context = ExecutionContext(
            # Execution identity
            user_id=user_id,
            graph_id=graph_id,
            graph_exec_id=graph_exec.id,
            graph_version=graph_exec.graph_version,
            # Safety settings
            human_in_the_loop_safe_mode=settings.human_in_the_loop_safe_mode,
            sensitive_action_safe_mode=settings.sensitive_action_safe_mode,
            dry_run=dry_run,
            # User settings
            user_timezone=(
                user.timezone if user.timezone != USER_TIMEZONE_NOT_SET else "UTC"
            ),
            # Execution hierarchy
            root_execution_id=graph_exec.id,
            # Workspace (enables workspace:// file resolution in blocks)
            workspace_id=workspace.id,
        )

    try:
        graph_exec_entry = graph_exec.to_graph_execution_entry(
            compiled_nodes_input_masks=compiled_nodes_input_masks,
            nodes_to_skip=nodes_to_skip,
            execution_context=execution_context,
        )
        logger.info(f"Queueing execution {graph_exec.id}")

        # Update execution status to QUEUED BEFORE publishing to prevent race condition
        # where two concurrent requests could both publish the same execution
        updated_exec = await edb.update_graph_execution_stats(
            graph_exec_id=graph_exec.id,
            status=ExecutionStatus.QUEUED,
        )

        # Verify the status update succeeded (prevents duplicate queueing in race conditions)
        # If another request already updated the status, this execution will not be QUEUED
        if not updated_exec or updated_exec.status != ExecutionStatus.QUEUED:
            logger.warning(
                f"Skipping queue publish for execution {graph_exec.id} - "
                f"status update failed or execution already queued by another request"
            )
            return graph_exec

        graph_exec.status = ExecutionStatus.QUEUED

        # Publish to execution queue for executor to pick up
        # This happens AFTER status update to ensure only one request publishes
        exec_queue = await get_async_execution_queue()
        await exec_queue.publish_message(
            routing_key=GRAPH_EXECUTION_ROUTING_KEY,
            message=graph_exec_entry.model_dump_json(),
            exchange=GRAPH_EXECUTION_EXCHANGE,
        )
        logger.info(f"Published execution {graph_exec.id} to RabbitMQ queue")
    except BaseException as e:
        err = str(e) or type(e).__name__
        if not graph_exec:
            logger.error(f"Unable to execute graph #{graph_id} failed: {err}")
            raise

        logger.error(
            f"Unable to publish graph #{graph_id} exec #{graph_exec.id}: {err}"
        )
        await edb.update_node_execution_status_batch(
            [node_exec.node_exec_id for node_exec in graph_exec.node_executions],
            ExecutionStatus.FAILED,
        )
        await edb.update_graph_execution_stats(
            graph_exec_id=graph_exec.id,
            status=ExecutionStatus.FAILED,
            stats=GraphExecutionStats(error=err),
        )
        raise

    try:
        await get_async_execution_event_bus().publish(graph_exec)
        logger.info(f"Published update for execution #{graph_exec.id} to event bus")
    except Exception as e:
        logger.error(
            f"Failed to publish execution event for graph exec #{graph_exec.id}: {e}"
        )

    try:
        await odb.increment_onboarding_runs(user_id)
        logger.info(
            f"Incremented user #{user_id} onboarding runs for exec #{graph_exec.id}"
        )
    except Exception as e:
        logger.error(f"Failed to increment onboarding runs for user #{user_id}: {e}")

    return graph_exec


# ============ Execution Output Helpers ============ #


class ExecutionOutputEntry(BaseModel):
    node: Node
    node_exec_id: str
    data: BlockOutputEntry


class NodeExecutionProgress:
    def __init__(self):
        self.output: dict[str, list[ExecutionOutputEntry]] = defaultdict(list)
        self.tasks: dict[str, Future] = {}
        self._lock = threading.Lock()

    def add_task(self, node_exec_id: str, task: Future):
        self.tasks[node_exec_id] = task

    def add_output(self, output: ExecutionOutputEntry):
        with self._lock:
            self.output[output.node_exec_id].append(output)

    def pop_output(self) -> ExecutionOutputEntry | None:
        exec_id = self._next_exec()
        if not exec_id:
            return None

        if self._pop_done_task(exec_id):
            return self.pop_output()

        with self._lock:
            if next_output := self.output[exec_id]:
                return next_output.pop(0)

        return None

    def is_done(self, wait_time: float = 0.0) -> bool:
        exec_id = self._next_exec()
        if not exec_id:
            return True

        if self._pop_done_task(exec_id):
            return self.is_done(wait_time)

        if wait_time <= 0:
            return False

        try:
            self.tasks[exec_id].result(wait_time)
        except TimeoutError:
            pass
        except Exception as e:
            logger.error(
                f"Task for exec ID {exec_id} failed with error: {e.__class__.__name__} {str(e)}"
            )
            pass
        return self.is_done(0)

    def stop(self) -> list[str]:
        """
        Stops all tasks and clears the output.
        This is useful for cleaning up when the execution is cancelled or terminated.
        Returns a list of execution IDs that were stopped.
        """
        cancelled_ids = []
        for task_id, task in self.tasks.items():
            if task.done():
                continue
            task.cancel()
            cancelled_ids.append(task_id)
        return cancelled_ids

    def wait_for_done(self, timeout: float = 5.0):
        """
        Wait for all cancelled tasks to complete cancellation.

        Args:
            timeout: Maximum time to wait for cancellation in seconds
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            while self.pop_output():
                pass

            if self.is_done():
                return

            time.sleep(0.1)  # Small delay to avoid busy waiting

        raise TimeoutError(
            f"Timeout waiting for cancellation of tasks: {list(self.tasks.keys())}"
        )

    def _pop_done_task(self, exec_id: str) -> bool:
        task = self.tasks.get(exec_id)
        if not task:
            return True

        if not task.done():
            return False

        with self._lock:
            if self.output[exec_id]:
                return False

        self.tasks.pop(exec_id)
        return True

    def _next_exec(self) -> str | None:
        if not self.tasks:
            return None
        return next(iter(self.tasks.keys()))
