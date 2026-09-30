"""Capy automations: standing jobs that start agent runs on a schedule or an event.

Each run is an ordinary Capy thread, so the thread blocks follow a run once it
has started. Automations get their own client module because they are a
separate API resource.
"""

import uuid
from typing import Any, Optional

from ._api import CapyClient, model_selection
from ._types import Automation, AutomationTrigger

# The events each event-driven trigger accepts, per Capy's API reference.
TRIGGER_EVENTS: dict[AutomationTrigger, tuple[str, ...]] = {
    AutomationTrigger.GITHUB: (
        "pull_request_opened",
        "pull_request_pushed",
        "pull_request_merged",
        "draft_opened",
        "checks",
        "workflow_run",
        "comment",
        "issue_comment",
        "pull_request_review_comment",
        "pull_request_review_submitted",
        "pull_request_review_thread",
        "branch_push",
        "label_change",
    ),
    AutomationTrigger.SLACK: ("message", "reaction", "channel_created"),
    AutomationTrigger.SENTRY: ("any_issue", "issue_lifecycle", "event_alert"),
    AutomationTrigger.LINEAR: ("issue_created", "status_changed", "end_cycle"),
}


class CapyAutomationsClient(CapyClient):
    async def list_automations(
        self, project_id: str, limit: int, cursor: str = ""
    ) -> tuple[list[Automation], Optional[str]]:
        data = await self._request(
            "GET",
            "/automations",
            params={"projectId": project_id, "limit": limit, "cursor": cursor},
        )
        items = [Automation.model_validate(a) for a in data.get("items", [])]
        return items, data.get("cursor")

    async def create_automation(
        self,
        *,
        project_id: str,
        name: str,
        prompt: str,
        trigger: dict[str, Any],
        description: str = "",
        model_id: str = "",
        reasoning: str = "",
        thread_mode: str = "",
        max_runs_per_day: Optional[int] = None,
        machine_size: str = "",
        enabled: bool = True,
        request_id: str = "",
    ) -> Automation:
        body: dict[str, Any] = {
            # Capy dedupes creates on requestId, so a retried block call can
            # never set up the same automation twice.
            "requestId": request_id or str(uuid.uuid4()),
            "projectId": project_id,
            "name": name,
            "prompt": prompt,
            "triggers": [trigger],
            "enabled": enabled,
        }
        if description:
            body["description"] = description
        if model := model_selection(model_id, reasoning):
            body["model"] = model
        if thread_mode:
            body["threadMode"] = thread_mode
        if max_runs_per_day is not None:
            body["maxRunsPerDay"] = max_runs_per_day
        if machine_size:
            body["machineSize"] = machine_size
        return Automation.model_validate(
            await self._request("POST", "/automations", body=body)
        )

    async def set_automation_enabled(
        self, automation_id: str, enabled: bool
    ) -> Automation:
        action = "enable" if enabled else "disable"
        return Automation.model_validate(
            await self._request("POST", f"/automations/{automation_id}/{action}")
        )

    async def delete_automation(self, automation_id: str) -> Automation:
        return Automation.model_validate(
            await self._request("POST", f"/automations/{automation_id}/delete")
        )


def trigger_payload(
    kind: AutomationTrigger,
    *,
    event: str = "",
    cron: str = "",
    timezone: str = "",
    conditions: Optional[dict[str, Any]] = None,
    run_when: str = "",
) -> dict[str, Any]:
    """The trigger as Capy's API takes it; a ValueError names what is missing.

    Schedule and on-demand triggers take no filters, so their conditions and
    run_when are left out rather than sent for Capy to reject.
    """
    if kind is AutomationTrigger.ON_DEMAND:
        return {"type": kind.value}
    if kind is AutomationTrigger.SCHEDULE:
        if not cron:
            raise ValueError(
                "A schedule trigger needs cron, e.g. '0 9 * * 1-5' for 09:00 "
                "on weekdays"
            )
        return {"type": kind.value, "cron": cron, "timezone": timezone or "UTC"}
    payload: dict[str, Any] = {"type": kind.value}
    if events := TRIGGER_EVENTS.get(kind):
        if event not in events:
            raise ValueError(
                f"A {kind.value} trigger needs event set to one of: "
                f"{', '.join(events)}"
            )
        payload["event"] = event
    if conditions:
        payload["conditions"] = conditions
    if run_when:
        payload["run_when"] = run_when
    return payload
