"""Async client for the All Quiet Public API v1."""

from typing import Any, Optional

from backend.sdk import APIKeyCredentials, Requests, json

from ._types import (
    AllQuietRegion,
    Incident,
    IncidentSeverity,
    IncidentSortBy,
    IncidentStatus,
    OnCallShift,
    Team,
)

REGION_HOSTS: dict[AllQuietRegion, str] = {
    AllQuietRegion.US: "https://allquiet.app",
    AllQuietRegion.EU: "https://allquiet.eu",
}


class AllQuietClient:
    """Thin wrapper over the All Quiet Public API.

    Responses are validated into the models in ``_types`` so blocks never hand
    raw dicts to downstream nodes.
    """

    def __init__(
        self,
        credentials: APIKeyCredentials,
        region: AllQuietRegion = AllQuietRegion.US,
    ):
        self.host = REGION_HOSTS[region]
        self.base_url = f"{self.host}/api/public/v1"
        self.requests = Requests(
            trusted_origins=list(REGION_HOSTS.values()),
            raise_for_status=False,
            extra_headers={
                "X-Api-Key": credentials.api_key.get_secret_value(),
                "Content-Type": "application/json",
            },
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        body: Optional[dict[str, Any]] = None,
    ) -> Any:
        response = await self.requests.request(
            method,
            f"{self.base_url}{path}",
            params=_clean_params(params or {}),
            json=body,
        )
        if not response.ok:
            raise RuntimeError(_error_message(response.status, response.text()))
        if not response.text():
            return None
        return response.json()

    async def create_incident(
        self,
        *,
        title: str,
        status: IncidentStatus,
        severity: IncidentSeverity,
        message: str = "",
        message_is_public: bool = False,
        team_ids: Optional[list[str]] = None,
        service_ids: Optional[list[str]] = None,
        user_ids: Optional[list[str]] = None,
        attributes: Optional[dict[str, str]] = None,
    ) -> Incident:
        payload: dict[str, Any] = {
            "title": title,
            "status": status.value,
            "severity": severity.value,
        }
        if message:
            payload["message"] = message
            payload["messageIsPublic"] = message_is_public
        if team_ids:
            payload["teamIds"] = team_ids
        if service_ids:
            payload["serviceIds"] = service_ids
        if user_ids:
            payload["userAssignments"] = [{"userId": uid} for uid in user_ids]
        if attributes:
            payload["attributes"] = [
                {"name": name, "value": value} for name, value in attributes.items()
            ]

        data = await self._request("POST", "/incident", body=payload)
        return _to_incident(data)

    async def get_incident(self, incident_id: str) -> Incident:
        data = await self._request("GET", f"/incident/search/{incident_id}")
        return _to_incident(data)

    async def get_incident_markdown(self, incident_id: str) -> str:
        response = await self.requests.get(
            f"{self.base_url}/incident/search/{incident_id}/markdown"
        )
        if not response.ok:
            raise RuntimeError(_error_message(response.status, response.text()))
        return response.text()

    async def list_incidents(
        self,
        *,
        statuses: Optional[list[IncidentStatus]] = None,
        severities: Optional[list[IncidentSeverity]] = None,
        team_ids: Optional[list[str]] = None,
        user_ids: Optional[list[str]] = None,
        search_term: str = "",
        unattended: Optional[bool] = None,
        created_from: str = "",
        created_until: str = "",
        limit: int = 25,
        offset: int = 0,
        sort_by: IncidentSortBy = IncidentSortBy.CREATED,
        ascending: bool = False,
    ) -> tuple[list[Incident], bool]:
        """Return the matching incidents and whether more pages are available."""
        params: dict[str, Any] = {
            "Statuses": [s.value for s in statuses or []],
            "Severities": [s.value for s in severities or []],
            "TeamIds": team_ids or [],
            "UserIds": user_ids or [],
            "SearchTerm": search_term,
            "Unattended": _bool_param(unattended),
            "CreatedFrom": created_from,
            "CreatedUntil": created_until,
            "Limit": limit,
            "Offset": offset,
            "SortBy": sort_by.value,
            "Asc": _bool_param(ascending),
        }
        data = await self._request("GET", "/incident/search/list", params=params)
        incidents = [_to_incident(item) for item in (data or {}).get("incidents", [])]
        return incidents, bool((data or {}).get("hasMore", False))

    async def append_intent(
        self,
        incident_id: str,
        *,
        intent: str,
        message: str = "",
        message_is_public: bool = False,
        severity: Optional[IncidentSeverity] = None,
    ) -> None:
        """Apply an intent (and optionally a new severity) to an incident.

        The patch endpoint returns a list payload rather than the updated
        incident, so callers re-read the incident to observe the new state.
        """
        operations: dict[str, Any] = {
            "appendIntent": {
                "intent": intent,
                "message": message,
                "messageIsPublic": message_is_public,
            }
        }
        if severity is not None:
            operations["changeSeverity"] = {"severity": severity.value}

        await self._request(
            "PATCH", f"/incident/{incident_id}", body={"operations": operations}
        )

    async def get_on_call(
        self,
        *,
        team_ids: Optional[list[str]] = None,
        user_ids: Optional[list[str]] = None,
        timestamp: str = "",
    ) -> list[OnCallShift]:
        params = {
            "TeamIds": team_ids or [],
            "UserIds": user_ids or [],
            "Timestamp": timestamp,
        }
        data = await self._request("GET", "/on-call", params=params)
        return [
            OnCallShift.model_validate(m) for m in (data or {}).get("memberships", [])
        ]

    async def list_teams(
        self,
        *,
        display_name: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Team], bool]:
        params = {"DisplayName": display_name, "Limit": limit, "Offset": offset}
        data = await self._request("GET", "/team/search/list", params=params)
        teams = [Team.model_validate(t) for t in (data or {}).get("teams", [])]
        return teams, bool((data or {}).get("hasMore", False))


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    """Drop unset params so All Quiet does not reject empty filter values."""
    return {
        key: value
        for key, value in params.items()
        if value is not None and value != "" and value != []
    }


def _bool_param(value: Optional[bool]) -> Optional[str]:
    """Render an optional flag as a query value.

    aiohttp's URL layer refuses raw ``bool`` query values, and All Quiet expects
    the lowercase spellings, so flags travel as ``"true"``/``"false"`` strings.
    ``None`` is passed through for ``_clean_params`` to drop.
    """
    if value is None:
        return None
    return "true" if value else "false"


def _to_incident(data: Optional[dict[str, Any]]) -> Incident:
    """Validate an incident payload, lifting status/severity off the latest event.

    All Quiet models an incident's current state as the head of its event
    timeline (newest first), so flatten it onto the incident for consumers.
    """
    if not data:
        raise RuntimeError("All Quiet returned an empty incident payload")

    incident = Incident.model_validate(data)
    events = data.get("events") or []
    if events:
        latest = events[0]
        if latest.get("status"):
            incident.status = IncidentStatus(latest["status"])
        if latest.get("severity"):
            incident.severity = IncidentSeverity(latest["severity"])
    return incident


def _error_message(status: int, body: str) -> str:
    """Turn All Quiet's RFC 9110 problem details into a single readable line."""
    detail = body.strip()
    parsed = json.loads(body, fallback={}) if body else {}

    if isinstance(parsed, dict):
        field_errors = parsed.get("errors")
        if isinstance(field_errors, dict):
            detail = "; ".join(
                f"{field}: {' '.join(messages)}"
                for field, messages in field_errors.items()
            )
        elif parsed.get("title"):
            detail = str(parsed["title"])

    if status == 401 or status == 403:
        return (
            f"All Quiet rejected the API key ({status}). Check the key is valid and "
            f"that your plan includes the Public API. {detail}".strip()
        )
    return f"All Quiet API error {status}: {detail}"
