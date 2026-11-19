"""Keep the ids an MCP call names inside the organization it acts in.

The Copilot tools look rows up by user. A v2 credential is bound to one
organization, and the REST API answers 404 for a row tagged with another
(`tenancy.in_tenant`); this gives an MCP call the same answer, before the tool
runs. Tools that list or search on their own filter what they return through
`copilot.tools.external_scope`.

Every argument of an exposed tool that looks like an id is classified below;
mcp_tenancy_test.py fails on one that isn't.
"""

import logging
from enum import Enum
from typing import Any

from mcp.server.fastmcp.exceptions import ToolError

from backend.copilot.tools.external_scope import in_tenant
from backend.data.db_accessors import execution_db, graph_db, library_db
from backend.util.exceptions import NotFoundError
from backend.util.text_limits import MAX_LIST_ENTRIES

logger = logging.getLogger(__name__)


class IdKind(Enum):
    LIBRARY_AGENT = "Library agent"
    FOLDER = "Folder"
    RUN = "Run"
    PRESET = "Preset"


AGENT = IdKind.LIBRARY_AGENT
FOLDER = IdKind.FOLDER
RUN = IdKind.RUN
PRESET = IdKind.PRESET

# Per tool, the arguments that name an organization's row. An agent id may be
# a library agent id or a graph id, as the tools accept either.
TENANTED_ARGS: dict[str, dict[str, IdKind]] = {
    "create_agent": {"library_agent_ids": AGENT, "folder_id": FOLDER},
    "customize_agent": {"library_agent_ids": AGENT, "folder_id": FOLDER},
    "edit_agent": {"agent_id": AGENT, "library_agent_ids": AGENT},
    "find_library_agent": {"agent_id": AGENT},
    "run_agent": {"library_agent_id": AGENT},
    "view_agent_output": {"library_agent_id": AGENT, "execution_id": RUN},
    "list_schedules": {"library_agent_id": AGENT},
    "create_folder": {"parent_id": FOLDER},
    "list_folders": {"parent_id": FOLDER},
    "update_folder": {"folder_id": FOLDER},
    "delete_folder": {"folder_id": FOLDER},
    "move_folder": {"folder_id": FOLDER, "target_parent_id": FOLDER},
    "move_agents_to_folder": {"agent_ids": AGENT, "folder_id": FOLDER},
    "list_presets": {"library_agent_id": AGENT, "graph_id": AGENT},
    "update_preset": {"preset_id": PRESET},
    "delete_preset": {"preset_id": PRESET},
    "list_agent_triggers": {"library_agent_id": AGENT},
    "setup_agent_webhook_trigger": {"library_agent_id": AGENT, "graph_id": AGENT},
}

# Arguments that look like ids but name no organization's row, with the reason.
UNTENANTED_ARGS: dict[str, dict[str, str]] = {
    "delete_schedule": {
        "schedule_id": "the schedule tools only find the organization's schedules",
    },
    "list_schedules": {
        "graph_id": "the schedule tools only list the organization's schedules",
    },
    "read_workspace_file": {"file_id": "workspace files belong to the user"},
    "delete_workspace_file": {"file_id": "workspace files belong to the user"},
    "list_workspace_files": {"folder_id": "a workspace folder, the user's own"},
    "run_agent": {"username_agent_slug": "names a public marketplace listing"},
    "view_agent_output": {"store_slug": "names a public marketplace listing"},
}


async def check_ids_in_tenant(
    tool_name: str, args: dict[str, Any], user_id: str, organization_id: str
) -> None:
    """Refuse a call naming a row tagged with another organization.

    An id that names nothing passes; the tool says it wasn't found. An argument
    of another type or length than the tool takes is refused before any lookup.
    """
    tenanted = TENANTED_ARGS.get(tool_name, {})
    for arg in tenanted:
        _check_shape(arg, args.get(arg))

    for arg, kind in tenanted.items():
        values = _ids(args.get(arg))
        if not values:
            continue
        try:
            organizations = await _organizations(kind, values, user_id)
        except Exception as exc:
            logger.exception(f"MCP tenancy check failed on {tool_name}.{arg}")
            raise ToolError("Couldn't check access; retry shortly") from exc
        for value in values:
            if not all(
                in_tenant(org, organization_id) for org in organizations.get(value, [])
            ):
                raise ToolError(f"{kind.value} '{value}' not found")


def _check_shape(arg: str, value: Any) -> None:
    if value is None:
        return
    if not arg.endswith("_ids"):
        if not isinstance(value, str):
            raise ToolError(f"{arg} must be a single id")
        return
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ToolError(f"{arg} must be a list of ids")
    if len(value) > MAX_LIST_ENTRIES:
        raise ToolError(f"{arg} takes at most {MAX_LIST_ENTRIES} ids")


def _ids(value: Any) -> list[str]:
    """The ids an argument holds, each once, normalised the way the tools read them."""
    values = value if isinstance(value, list) else [value]
    return list(
        dict.fromkeys(v.strip() for v in values if isinstance(v, str) and v.strip())
    )


async def _organizations(
    kind: IdKind, values: list[str], user_id: str
) -> dict[str, list[str | None]]:
    """For each id, the organization of every row it could name."""
    match kind:
        case IdKind.LIBRARY_AGENT:
            return await _agent_organizations(values, user_id)
        case IdKind.FOLDER:
            return {v: await _folder_organizations(v, user_id) for v in values}
        case IdKind.RUN:
            return {v: await _run_organizations(v, user_id) for v in values}
        case IdKind.PRESET:
            return {v: await _preset_organizations(v, user_id) for v in values}


async def _agent_organizations(
    values: list[str], user_id: str
) -> dict[str, list[str | None]]:
    """The organization of every row each id could name, as the tools resolve it.

    A library agent by its own id or by its graph's; failing both, a graph of
    the user's own with no library entry, which ``edit_agent`` opens directly.
    Two reads of the tags alone, however many ids are listed.
    """
    organizations = await library_db().get_library_agent_organizations_by_any_id(
        user_id, values
    )
    if unlisted := [value for value in values if value not in organizations]:
        owned = await graph_db().get_owned_graph_organizations(user_id, unlisted)
        organizations.update({graph_id: [org] for graph_id, org in owned.items()})
    return organizations


async def _folder_organizations(value: str, user_id: str) -> list[str | None]:
    try:
        folder = await library_db().get_folder(value, user_id)
    except NotFoundError:
        return []
    return [folder.organization_id]


async def _run_organizations(value: str, user_id: str) -> list[str | None]:
    run = await execution_db().get_graph_execution_meta(
        user_id=user_id, execution_id=value
    )
    return [run.organization_id] if run else []


async def _preset_organizations(value: str, user_id: str) -> list[str | None]:
    preset = await library_db().get_preset(user_id=user_id, preset_id=value)
    return [preset.organization_id] if preset else []
