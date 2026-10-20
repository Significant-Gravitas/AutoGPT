"""Unit tests for the All Quiet API client's request shaping and error handling."""

from typing import Any, Optional
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from backend.blocks.allquiet._api import (
    REGION_HOSTS,
    AllQuietClient,
    _bool_param,
    _clean_params,
    _error_message,
    _to_incident,
)
from backend.blocks.allquiet._types import (
    AllQuietRegion,
    IncidentSeverity,
    IncidentStatus,
)
from backend.data.model import APIKeyCredentials


def _credentials() -> APIKeyCredentials:
    return APIKeyCredentials(
        id="01234567-89ab-cdef-0123-456789abcdef",
        provider="allquiet",
        api_key=SecretStr("test-key"),
        title="Test key",
        expires_at=None,
    )


class _FakeResponse:
    def __init__(self, status: int = 200, payload: Any = None, body: str = ""):
        self.status = status
        self._payload = payload
        self._body = body

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> Any:
        if self._payload is None and self._body:
            raise ValueError("not JSON")
        return self._payload

    def text(self) -> str:
        if self._body:
            return self._body
        return "{}" if self._payload is not None else ""


def _client_with(response: _FakeResponse) -> tuple[AllQuietClient, AsyncMock]:
    """Build a client whose HTTP layer is swapped for a recording mock."""
    client = AllQuietClient(_credentials())
    request = AsyncMock(return_value=response)
    # Patch the transport itself so the tests cover the client's own request
    # shaping (params, body, error handling) without any network access.
    requests: Any = client.requests
    requests.request = request
    return client, request


def _sent(request: AsyncMock) -> tuple[str, str, dict[str, Any], Optional[dict]]:
    method, url = request.await_args.args
    kwargs = request.await_args.kwargs
    return method, url, kwargs.get("params") or {}, kwargs.get("json")


def _incident_payload(status: str = "Open", severity: str = "Critical") -> dict:
    return {
        "id": "inc-1",
        "title": "Checkout latency above SLO",
        "createdAt": "2026-08-16T23:42:17.274Z",
        "lastUpdatedAt": "2026-08-16T23:57:59.023Z",
        "allowedIntents": ["Investigated", "Resolved"],
        "events": [
            {"status": status, "severity": severity},
            {"status": "Open", "severity": "Minor"},
        ],
    }


class TestRegions:
    def test_us_is_the_default(self):
        client = AllQuietClient(_credentials())
        assert client.base_url == "https://allquiet.app/api/public/v1"

    def test_eu_region_targets_the_eu_host(self):
        client = AllQuietClient(_credentials(), AllQuietRegion.EU)
        assert client.base_url == "https://allquiet.eu/api/public/v1"

    def test_both_hosts_are_trusted_origins(self):
        client = AllQuietClient(_credentials())
        for host in REGION_HOSTS.values():
            assert host.removeprefix("https://") in client.requests.trusted_origins


class TestCleanParams:
    def test_drops_unset_values(self):
        assert _clean_params({"a": None, "b": "", "c": []}) == {}

    def test_keeps_zero(self):
        # `Offset=0` is a meaningful value, not an absent one.
        assert _clean_params({"Offset": 0}) == {"Offset": 0}


class TestBoolParam:
    def test_renders_lowercase_strings(self):
        # aiohttp's URL layer raises TypeError on raw bool query values, so
        # flags must reach it already rendered as strings.
        assert _bool_param(True) == "true"
        assert _bool_param(False) == "false"

    def test_passes_none_through_for_cleaning(self):
        assert _bool_param(None) is None


class TestToIncident:
    def test_lifts_status_and_severity_off_the_newest_event(self):
        incident = _to_incident(_incident_payload("Resolved", "Warning"))

        assert incident.status == IncidentStatus.RESOLVED
        assert incident.severity == IncidentSeverity.WARNING

    def test_maps_camel_case_fields(self):
        incident = _to_incident(_incident_payload())

        assert incident.created_at == "2026-08-16T23:42:17.274Z"
        assert incident.allowed_intents == ["Investigated", "Resolved"]

    def test_tolerates_an_incident_with_no_events(self):
        payload = _incident_payload()
        payload["events"] = []

        incident = _to_incident(payload)

        assert incident.status is None
        assert incident.severity is None

    def test_rejects_an_empty_payload(self):
        with pytest.raises(RuntimeError, match="empty incident payload"):
            _to_incident(None)


class TestErrorMessage:
    def test_flattens_field_validation_errors(self):
        body = (
            '{"title":"One or more validation errors occurred.","status":400,'
            '"errors":{"Statuses":["Values must be one of Open,Resolved"]}}'
        )

        message = _error_message(400, body)

        assert "Statuses: Values must be one of Open,Resolved" in message
        assert "400" in message

    def test_calls_out_a_bad_key_on_401(self):
        message = _error_message(401, "")

        assert "rejected the API key" in message

    def test_falls_back_to_the_raw_body_when_not_json(self):
        assert "upstream exploded" in _error_message(500, "upstream exploded")


class TestCreateIncident:
    async def test_sends_only_the_fields_that_were_set(self):
        client, request = _client_with(_FakeResponse(payload=_incident_payload()))

        await client.create_incident(
            title="Checkout latency above SLO",
            status=IncidentStatus.OPEN,
            severity=IncidentSeverity.CRITICAL,
        )

        _, _, _, body = _sent(request)
        assert body == {
            "title": "Checkout latency above SLO",
            "status": "Open",
            "severity": "Critical",
        }

    async def test_maps_attributes_and_user_assignments(self):
        client, request = _client_with(_FakeResponse(payload=_incident_payload()))

        await client.create_incident(
            title="t",
            status=IncidentStatus.OPEN,
            severity=IncidentSeverity.MINOR,
            user_ids=["u1"],
            attributes={"host": "web-01"},
        )

        _, _, _, body = _sent(request)
        assert body is not None
        assert body["userAssignments"] == [{"userId": "u1"}]
        assert body["attributes"] == [{"name": "host", "value": "web-01"}]

    async def test_raises_a_readable_error_on_failure(self):
        client, _ = _client_with(
            _FakeResponse(status=400, body='{"title":"Bad request"}')
        )

        with pytest.raises(RuntimeError, match="Bad request"):
            await client.create_incident(
                title="t",
                status=IncidentStatus.OPEN,
                severity=IncidentSeverity.MINOR,
            )


class TestListIncidents:
    async def test_serialises_enum_filters_and_reports_paging(self):
        client, request = _client_with(
            _FakeResponse(payload={"incidents": [_incident_payload()], "hasMore": True})
        )

        incidents, has_more = await client.list_incidents(
            statuses=[IncidentStatus.OPEN],
            severities=[IncidentSeverity.CRITICAL, IncidentSeverity.WARNING],
            limit=10,
        )

        _, _, params, _ = _sent(request)
        assert params["Statuses"] == ["Open"]
        assert params["Severities"] == ["Critical", "Warning"]
        assert len(incidents) == 1
        assert has_more is True

    async def test_omits_empty_filters(self):
        client, request = _client_with(_FakeResponse(payload={"incidents": []}))

        await client.list_incidents()

        _, _, params, _ = _sent(request)
        assert "Statuses" not in params
        assert "SearchTerm" not in params

    async def test_sends_flags_as_strings_not_bools(self):
        # Regression: raw bools reach yarl and raise
        # "Invalid variable type: value should be str, int or float".
        client, request = _client_with(_FakeResponse(payload={"incidents": []}))

        await client.list_incidents(unattended=True, ascending=False)

        _, _, params, _ = _sent(request)
        assert params["Unattended"] == "true"
        assert params["Asc"] == "false"
        assert not any(value is True or value is False for value in params.values())

    async def test_omits_the_unattended_filter_when_unset(self):
        client, request = _client_with(_FakeResponse(payload={"incidents": []}))

        await client.list_incidents()

        _, _, params, _ = _sent(request)
        assert "Unattended" not in params


class TestAppendIntent:
    async def test_wraps_the_intent_in_an_operations_object(self):
        client, request = _client_with(_FakeResponse(payload={"incidents": []}))

        await client.append_intent("inc-1", intent="Resolved", message="fixed")

        method, url, _, body = _sent(request)
        assert method == "PATCH"
        assert url.endswith("/incident/inc-1")
        assert body == {
            "operations": {
                "appendIntent": {
                    "intent": "Resolved",
                    "message": "fixed",
                    "messageIsPublic": False,
                }
            }
        }

    async def test_adds_a_severity_change_when_requested(self):
        client, request = _client_with(_FakeResponse(payload={"incidents": []}))

        await client.append_intent(
            "inc-1", intent="Commented", severity=IncidentSeverity.CRITICAL
        )

        _, _, _, body = _sent(request)
        assert body is not None
        assert body["operations"]["changeSeverity"] == {"severity": "Critical"}


class TestOnCallAndTeams:
    async def test_parses_on_call_memberships(self):
        client, _ = _client_with(
            _FakeResponse(
                payload={
                    "memberships": [
                        {
                            "user": {"id": "u1", "displayName": "Ada"},
                            "team": {"id": "t1", "displayName": "Platform"},
                            "availabilities": [{"tier": 1, "isOnline": True}],
                        }
                    ]
                }
            )
        )

        shifts = await client.get_on_call()

        assert len(shifts) == 1
        assert shifts[0].user is not None and shifts[0].user.display_name == "Ada"
        assert shifts[0].availabilities[0].is_online is True

    async def test_parses_teams(self):
        client, _ = _client_with(
            _FakeResponse(
                payload={
                    "teams": [
                        {"id": "t1", "displayName": "Platform", "timeZoneId": "UTC"}
                    ],
                    "hasMore": False,
                }
            )
        )

        teams, has_more = await client.list_teams()

        assert teams[0].display_name == "Platform"
        assert teams[0].time_zone_id == "UTC"
        assert has_more is False


class TestListShapedErrors:
    """All Quiet returns `errors` as a list for auth failures, not a dict."""

    def test_flattens_the_list_envelope_used_by_auth_failures(self):
        # Captured verbatim from a real 401 against the live API.
        body = (
            '{"succeeded":false,'
            '"errors":[{"description":"Provided API key is invalid."}]}'
        )

        message = _error_message(401, body)

        assert "Provided API key is invalid." in message
        assert "rejected the API key" in message

    def test_joins_multiple_listed_errors(self):
        body = '{"errors":[{"description":"first"},{"description":"second"}]}'

        assert "first; second" in _error_message(400, body)

    def test_still_flattens_the_dict_envelope_used_by_validation(self):
        body = '{"errors":{"Statuses":["Values must be one of Open,Resolved"]}}'

        assert "Statuses: Values must be one of Open,Resolved" in _error_message(
            400, body
        )

    def test_falls_back_to_title_when_errors_is_unusable(self):
        body = '{"title":"One or more validation errors occurred.","errors":42}'

        assert "One or more validation errors occurred." in _error_message(400, body)

    def test_falls_back_to_the_raw_body_for_a_non_object_payload(self):
        assert "just a string" in _error_message(500, '"just a string"')


class TestGetIncidentMarkdown:
    async def test_returns_the_markdown_report(self):
        report = "# Checkout latency above SLO\n\n- **Status**: Open\n"
        client = AllQuietClient(_credentials())
        get = AsyncMock(return_value=_FakeResponse(body=report))
        requests: Any = client.requests
        requests.get = get

        assert await client.get_incident_markdown("inc-1") == report
        assert get.await_args.args[0].endswith("/incident/search/inc-1/markdown")

    async def test_raises_a_readable_error_when_the_report_fails(self):
        client = AllQuietClient(_credentials())
        requests: Any = client.requests
        requests.get = AsyncMock(
            return_value=_FakeResponse(status=404, body='{"title":"Not found"}')
        )

        with pytest.raises(RuntimeError, match="Not found"):
            await client.get_incident_markdown("nope")


class TestUnmodeledEnumValues:
    """Live event data may carry values beyond what the enums model."""

    def test_ignores_an_unmodeled_status_instead_of_raising(self):
        # Regression: bare IncidentStatus(...) coercion raised ValueError on
        # every read of such an incident, losing an otherwise usable payload.
        incident = _to_incident(_incident_payload(status="Acknowledged"))

        assert incident.status is None
        assert incident.title == "Checkout latency above SLO"

    def test_ignores_an_unmodeled_severity_instead_of_raising(self):
        incident = _to_incident(_incident_payload(severity="Fatal"))

        assert incident.severity is None
        assert incident.id == "inc-1"

    def test_still_maps_the_half_it_recognizes(self):
        incident = _to_incident(
            _incident_payload(status="Resolved", severity="Catastrophic")
        )

        assert incident.status == IncidentStatus.RESOLVED
        assert incident.severity is None


class TestNonJsonSuccessBody:
    async def test_a_2xx_with_a_non_json_body_raises_a_readable_error(self):
        # Guards against an unhandled JSONDecodeError crashing the block when a
        # proxy or the endpoint returns HTML on a 200.
        client, _ = _client_with(_FakeResponse(status=200, body="<html>oops</html>"))

        with pytest.raises(RuntimeError, match="not JSON"):
            await client.list_teams()


class TestEmptyErrorsObject:
    def test_falls_back_to_title_when_errors_is_empty(self):
        body = '{"title":"Something went wrong","errors":{}}'

        assert "Something went wrong" in _error_message(400, body)
