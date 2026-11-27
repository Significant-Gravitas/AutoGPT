"""Shared fix → validate → preview/save pipeline for agent tools."""

import json
import logging
from collections import Counter
from typing import Any, cast

from backend.api.features.library.exceptions import (
    LibraryAgentInAnotherOrganizationError,
)
from backend.copilot.tools.models import (
    AgentPreviewResponse,
    AgentSavedResponse,
    ErrorResponse,
    ToolResponseBase,
)
from backend.integrations.webhooks.graph_lifecycle_hooks import GraphActivationError
from backend.util.exceptions import NotAuthorizedError, NotFoundError

from .blocks import get_blocks_as_dicts
from .core import get_library_agents_by_ids, save_agent_to_library
from .fixer import AgentFixer
from .switched_off_models import SwitchedOffMove
from .validator import AgentValidator

logger = logging.getLogger(__name__)

# Kinds of move (block, old model, new model) the note lists before "and N more".
_HINT_MOVES = 5

MAX_AGENT_JSON_SIZE = 1_000_000  # 1 MB

# Raised with text written for whoever saves the agent. Any other exception's
# text can name a query, a table or a provider's response.
_CALLER_FACING_ERRORS = (
    GraphActivationError,
    LibraryAgentInAnotherOrganizationError,
    NotAuthorizedError,
    NotFoundError,
    ValueError,
)


def _error_text(error: Exception, external_caller: bool) -> str:
    """The exception's text, unless an External API client would see text
    that wasn't written for it."""
    if external_caller and not isinstance(error, _CALLER_FACING_ERRORS):
        return "an internal error"
    return str(error)


async def fetch_library_agents(
    user_id: str | None,
    library_agent_ids: list[str],
) -> list[dict[str, Any]] | None:
    """Fetch library agents by IDs for AgentExecutorBlock validation.

    Returns None if no IDs provided or user is not authenticated.
    """
    if not user_id or not library_agent_ids:
        return None
    try:
        agents = await get_library_agents_by_ids(
            user_id=user_id,
            agent_ids=library_agent_ids,
        )
        return cast(list[dict[str, Any]], agents)
    except Exception as e:
        logger.warning(f"Failed to fetch library agents by IDs: {e}")
        return None


def agent_json_too_large(
    agent_json: dict[str, Any], session_id: str | None
) -> ErrorResponse | None:
    """The error for a workflow over MAX_AGENT_JSON_SIZE, or None when it fits.
    The fixer runs on the event loop and costs more with every node, so both
    the save path and fix_agent_graph check this before it runs."""
    json_size = len(json.dumps(agent_json))
    if json_size <= MAX_AGENT_JSON_SIZE:
        return None
    return ErrorResponse(
        message=(
            f"This workflow is too large ({json_size:,} bytes, "
            f"max {MAX_AGENT_JSON_SIZE:,}). Reduce the number of nodes."
        ),
        error="agent_json_too_large",
        session_id=session_id,
    )


async def fix_validate_and_save(
    agent_json: dict[str, Any],
    *,
    user_id: str | None,
    session_id: str | None,
    save: bool = True,
    is_update: bool = False,
    default_name: str = "Workflow",
    preview_message: str | None = None,
    save_message: str | None = None,
    library_agents: list[dict[str, Any]] | None = None,
    folder_id: str | None = None,
    is_hidden: bool = False,
    organization_id: str | None = None,
    team_id: str | None = None,
    move_webhook_presets: bool = True,
    external_caller: bool = False,
    saved_agent: dict[str, Any] | None = None,
) -> ToolResponseBase:
    """Shared pipeline: auto-fix → validate → preview or save.

    Args:
        agent_json: The agent JSON dict (must already have id/version/is_active set).
        user_id: The authenticated user's ID.
        session_id: The chat session ID.
        save: Whether to save or just preview.
        is_update: Whether this is an update to an existing agent.
        default_name: Fallback name if agent_json has none.
        preview_message: Custom preview message (optional).
        save_message: Custom save success message (optional).
        library_agents: Library agents for AgentExecutorBlock validation/fixing.
        organization_id, team_id: Tenancy for a new graph or version; None
            keeps the defaults.
        move_webhook_presets: On an update, move the agent's webhook presets
            onto the new version.
        external_caller: An External API client is saving; a failed save
            tells it only what was written for it.
        saved_agent: The stored version an edit starts from: a node keeps a
            switched-off model it already has there. None for new work.

    Returns:
        An appropriate ToolResponseBase subclass.
    """
    if (too_large := agent_json_too_large(agent_json, session_id)) is not None:
        return too_large

    blocks = get_blocks_as_dicts()

    # Auto-fix
    fixer = AgentFixer()
    try:
        agent_json = fixer.apply_all_fixes(
            agent_json, blocks, library_agents, saved_agent=saved_agent
        )
        fixes = fixer.get_fixes_applied()
        if fixes:
            logger.info(f"Applied {len(fixes)} auto-fixes to agent JSON")
    except Exception as e:
        logger.warning(f"Auto-fix failed: {e}")
        # The fixes run in order, so one that raised before the model fix
        # skipped it, and a new node would be saved on a switched-off model.
        # Running it again is harmless: a node it already moved or kept
        # stays as it is.
        try:
            agent_json = fixer.fix_ai_model_parameter(
                agent_json, blocks, saved_models=fixer.saved_models
            )
        except Exception as model_fix_error:
            logger.warning(
                f"Model fix after a failed auto-fix failed: {model_fix_error}"
            )
    # The fixes edit agent_json in place, so a model moved before a later fix
    # raised stays moved, and the note must still say so.
    retired_models_hint = _retired_models_hint(fixer.retired_models_replaced)

    # Validate
    try:
        validator = AgentValidator()
        is_valid, _ = validator.validate(agent_json, blocks, library_agents)
        if not is_valid:
            errors = validator.errors
            return ErrorResponse(
                message=(
                    f"Validation failed with {len(errors)} error"
                    f"{'s' if len(errors) != 1 else ''}."
                ),
                error="validation_failed",
                details={"errors": errors},
                session_id=session_id,
            )
    except Exception as e:
        logger.error(f"Validation failed with exception: {e}", exc_info=True)
        return ErrorResponse(
            message="Failed to validate the workflow. Please try again.",
            error="validation_exception",
            details={"exception": _error_text(e, external_caller)},
            session_id=session_id,
        )

    agent_name = agent_json.get("name", default_name)
    agent_description = agent_json.get("description", "")
    node_count = len(agent_json.get("nodes", []))
    link_count = len(agent_json.get("links", []))

    # Build a warning suffix when name/description is missing or generic
    _GENERIC_NAMES = {
        "agent",
        "generated agent",
        "customized agent",
        "updated agent",
        "new agent",
        "my agent",
        "workflow",
        "generated workflow",
        "customized workflow",
        "updated workflow",
        "new workflow",
        "my workflow",
    }
    metadata_warnings: list[str] = []
    if not agent_json.get("name") or agent_name.lower().strip() in _GENERIC_NAMES:
        metadata_warnings.append("'name'")
    if not agent_description:
        metadata_warnings.append("'description'")
    metadata_hint = ""
    if metadata_warnings:
        missing = " and ".join(metadata_warnings)
        metadata_hint = (
            f" Note: the workflow is missing a meaningful {missing}. "
            f"Please update the agent_json to include them."
        )

    if not save:
        return AgentPreviewResponse(
            message=(
                (
                    preview_message
                    or f"Workflow '{agent_name}' with {node_count} blocks is ready."
                )
                + retired_models_hint
                + metadata_hint
            ),
            agent_json=agent_json,
            agent_name=agent_name,
            description=agent_description,
            node_count=node_count,
            link_count=link_count,
            session_id=session_id,
        )

    if not user_id:
        return ErrorResponse(
            message="You must be logged in to save workflows.",
            error="auth_required",
            session_id=session_id,
        )

    try:
        created_graph, library_agent = await save_agent_to_library(
            agent_json,
            user_id,
            is_update=is_update,
            folder_id=folder_id,
            is_hidden=is_hidden,
            organization_id=organization_id,
            team_id=team_id,
            move_webhook_presets=move_webhook_presets,
        )
        return AgentSavedResponse(
            message=(
                (save_message or f"Workflow '{created_graph.name}' has been saved!")
                + retired_models_hint
                + metadata_hint
            ),
            agent_id=created_graph.id,
            agent_name=created_graph.name,
            graph_version=created_graph.version,
            library_agent_id=library_agent.id,
            library_agent_link=f"/library/agents/{library_agent.id}",
            agent_page_link=f"/build?flowID={created_graph.id}",
            session_id=session_id,
        )
    except Exception as e:
        logger.error(f"Failed to save agent: {e}", exc_info=True)
        detail = _error_text(e, external_caller)
        return ErrorResponse(
            message=f"Failed to save the workflow: {detail}",
            error="save_failed",
            details={"exception": detail},
            session_id=session_id,
        )


def _retired_models_hint(moves: list[SwitchedOffMove]) -> str:
    """A note for Otto saying which blocks' nodes the fixer moved off a
    switched-off model and the model they use now, so Otto reports the model
    the workflow uses. It names blocks and catalog spellings only, never text
    from the nodes, and lists at most _HINT_MOVES kinds of move."""
    if not moves:
        return ""
    counts = Counter(moves)
    shown = [
        f"{move.block} uses '{move.new}' instead of '{move.old}' "
        f"({count} node{'' if count == 1 else 's'})"
        for move, count in list(counts.items())[:_HINT_MOVES]
    ]
    if len(counts) > _HINT_MOVES:
        shown.append(f"and {len(counts) - _HINT_MOVES} more")
    return (
        " Note: some nodes were on a switched-off model and now use another: "
        f"{'; '.join(shown)}. Tell the user which model these nodes use now."
    )
