from urllib.parse import quote, urlencode

PERSONAL_ORGANIZATION = "__personal__"
ORGANIZATION_HOME = "__org_home__"


def _tenant_params(
    organization_id: str | None,
    team_id: str | None,
) -> dict[str, str]:
    return {
        "organizationId": organization_id or PERSONAL_ORGANIZATION,
        "teamId": team_id or ORGANIZATION_HOME,
    }


def library_agent_path(
    library_agent_id: str,
    organization_id: str | None,
    team_id: str | None,
    *,
    active_tab: str | None = None,
    active_item: str | None = None,
) -> str:
    params = _tenant_params(organization_id, team_id)
    if active_tab:
        params["activeTab"] = active_tab
    if active_item:
        params["activeItem"] = active_item
    return f"/library/agents/{quote(library_agent_id, safe='')}?{urlencode(params)}"


def builder_path(
    graph_id: str,
    graph_version: int | None,
    organization_id: str | None,
    team_id: str | None,
    *,
    execution_id: str | None = None,
) -> str:
    params = _tenant_params(organization_id, team_id)
    params["flowID"] = graph_id
    if graph_version is not None:
        params["flowVersion"] = str(graph_version)
    if execution_id:
        params["flowExecutionID"] = execution_id
    return f"/build?{urlencode(params)}"


def copilot_path(
    session_id: str,
    organization_id: str | None,
    team_id: str | None,
) -> str:
    params = _tenant_params(organization_id, team_id)
    params["sessionId"] = session_id
    return f"/copilot?{urlencode(params)}"
