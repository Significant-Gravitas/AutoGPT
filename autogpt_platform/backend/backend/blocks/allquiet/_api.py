"""Async client for the All Quiet Public API v1."""

from typing import Any, Optional

from pydantic import BaseModel, TypeAdapter, ValidationError

from backend.sdk import APIKeyCredentials, Requests, json
from backend.util.request import Response

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


class _ListedError(BaseModel):
    """One entry of All Quiet's list-shaped ``errors`` envelope."""

    description: str = ""


_JSON_OBJECT = TypeAdapter(dict[str, Any])
_FIELD_ERRORS = TypeAdapter(dict[str, list[str]])
_LISTED_ERRORS = TypeAdapter(list[_ListedError])

# Membership sets for defensive coercion of live event data.
_STATUS_VALUES = {member.value for member in IncidentStatus}
_SEVERITY_VALUES = {member.value for member in IncidentSeverity}


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
        body_text = _raise_for_status(response)
        if not body_text:
            return None
        try:
            return response.json()
        except ValueError as exc:
            # A 2xx with a non-JSON body means the endpoint or a proxy in front
            # of it returned something unexpected; surface it as a readable
            # client error rather than an unhandled decode crash.
            raise RuntimeError(
                f"All Quiet returned a {response.status} with a body that is not "
                f"JSON: {body_text[:200]}"
            ) from exc

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
        return _raise_for_status(response)

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

        The intent is applied before that re-read, so a failed re-read reports
        an error even though the write landed — and a naive retry would apply
        the intent twice, leaving a duplicate Commented/Escalated entry on the
        timeline. Retry only after confirming the incident's current state.
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


def _raise_for_status(response: Response) -> str:
    """Return the body text, raising a readable error on any non-2xx status."""
    body_text = response.text()
    if not response.ok:
        raise RuntimeError(_error_message(response.status, body_text))
    return body_text


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
        # Coerce defensively: an unmodeled value would otherwise raise
        # ValueError on every read of that incident. The rest of the incident is
        # still useful, so leave the field unset rather than failing the call.
        # This mirrors the guard the trigger block already applies to payloads.
        if latest.get("status") in _STATUS_VALUES:
            incident.status = IncidentStatus(latest["status"])
        if latest.get("severity") in _SEVERITY_VALUES:
            incident.severity = IncidentSeverity(latest["severity"])
    return incident


def _error_message(status: int, body: str) -> str:
    """Turn an All Quiet error body into a single readable line."""
    detail = _error_detail(body) or body.strip()

    if status in (401, 403):
        return (
            f"All Quiet rejected the API key ({status}). Check the key is valid and "
            f"that your plan includes the Public API. {detail}".strip()
        )
    return f"All Quiet API error {status}: {detail}"


def _error_detail(body: str) -> str:
    """Extract the human-readable part of an All Quiet error body.

    All Quiet uses two different error shapes, so both are handled:

    * RFC 9110 problem details for request validation, where ``errors`` maps a
      field to its messages --
      ``{"errors": {"Statuses": ["Values must be one of Open,Resolved"]}}``
    * a result envelope for auth and similar failures, where ``errors`` is a
      list of objects --
      ``{"succeeded": false, "errors": [{"description": "Provided API key is invalid."}]}``

    Returns an empty string when nothing useful can be pulled out, so the caller
    can fall back to the raw body.
    """
    try:
        parsed = _JSON_OBJECT.validate_python(
            json.loads(body, fallback={}) if body else {}
        )
    except ValidationError:
        # Body parsed as JSON but isn't an object (e.g. a bare string or array).
        return ""
    if not parsed:
        return ""

    raw_errors = parsed.get("errors")

    try:
        field_errors = _FIELD_ERRORS.validate_python(raw_errors)
    except ValidationError:
        pass
    else:
        joined = "; ".join(
            f"{field}: {' '.join(messages)}" for field, messages in field_errors.items()
        )
        # An empty `errors` object validates but says nothing; fall through to
        # `title` rather than returning a blank detail.
        if joined:
            return joined

    try:
        listed_errors = _LISTED_ERRORS.validate_python(raw_errors)
    except ValidationError:
        pass
    else:
        described = [e.description for e in listed_errors if e.description]
        if described:
            return "; ".join(described)

    title = parsed.get("title")
    return str(title) if title else ""
