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

logger = logging.getLogger(__name__)


class IdKind(Enum):
    LIBRARY_AGENT = "Library agent"
    FOLDER = "Folder"
    RUN = "Run"


AGENT = IdKind.LIBRARY_AGENT
FOLDER = IdKind.FOLDER
RUN = IdKind.RUN

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

    An id that names nothing passes; the tool says it wasn't found.
    """
    for arg, kind in TENANTED_ARGS.get(tool_name, {}).items():
        for value in _ids(args.get(arg)):
            try:
                allowed = await _in_tenant(kind, value, user_id, organization_id)
            except Exception as exc:
                logger.exception(f"MCP tenancy check failed on {tool_name}.{arg}")
                raise ToolError("Couldn't check access; retry shortly") from exc
            if not allowed:
                raise ToolError(f"{kind.value} '{value}' not found")


def _ids(value: Any) -> list[str]:
    """The ids an argument holds, normalised the way the tools read them."""
    values = value if isinstance(value, list) else [value]
    return [v.strip() for v in values if isinstance(v, str) and v.strip()]


async def _in_tenant(
    kind: IdKind, value: str, user_id: str, organization_id: str
) -> bool:
    match kind:
        case IdKind.LIBRARY_AGENT:
            organizations = await _agent_organizations(value, user_id)
        case IdKind.FOLDER:
            organizations = await _folder_organizations(value, user_id)
        case IdKind.RUN:
            run = await execution_db().get_graph_execution_meta(
                user_id=user_id, execution_id=value
            )
            organizations = [run.organization_id] if run else []
    return all(in_tenant(org, organization_id) for org in organizations)


async def _agent_organizations(value: str, user_id: str) -> list[str | None]:
    """The organization of every row the id could name, as the tools resolve it.

    A library agent by its own id or by its graph's; failing both, a graph of
    the user's own with no library entry, which ``edit_agent`` opens directly.
    """
    lib = library_db()
    try:
        by_id = await lib.get_library_agent(value, user_id)
    except NotFoundError:
        by_id = None
    by_graph = await lib.get_library_agent_by_graph_id(
        user_id, value, include_archived=True
    )
    organizations = [a.organization_id for a in (by_id, by_graph) if a is not None]
    if organizations:
        return organizations
    graph = await graph_db().get_graph(value, None, user_id=user_id)
    if graph is not None and graph.user_id == user_id:
        return [graph.organization_id]
    return []


async def _folder_organizations(value: str, user_id: str) -> list[str | None]:
    try:
        folder = await library_db().get_folder(value, user_id)
    except NotFoundError:
        return []
    return [folder.organization_id]
