"""
Sample Exa Search Monitor responses, shaped like the API spec.

Used as block test mocks and as fixtures for the unit tests.
"""

SAMPLE_MONITOR_ID = "01k4d9w6y3h7p2m8n5q1r0s4tv"
SAMPLE_RUN_ID = "01k4dbq2x5j8r3n6m9p1s4t7vw"

SAMPLE_MONITOR = {
    "id": SAMPLE_MONITOR_ID,
    "name": "Battery recycling expansion",
    "status": "active",
    "search": {
        "query": "new battery recycling facilities announced in North America",
        "numResults": 10,
        "contents": {"highlights": True},
    },
    "trigger": {"type": "interval", "period": "1d"},
    "outputSchema": None,
    "metadata": {"workspace_id": "workspace_123"},
    "webhook": {
        "url": "https://example.com/webhooks/exa",
        "events": ["monitor.run.completed"],
    },
    "nextRunAt": "2026-10-29T20:00:00.000Z",
    "createdAt": "2026-10-28T20:00:00.000Z",
    "updatedAt": "2026-10-28T20:00:00.000Z",
}

SAMPLE_CREATED_MONITOR = {
    **SAMPLE_MONITOR,
    "webhookSecret": "whsec_sample_one_time_secret",  # pragma: allowlist secret
}

SAMPLE_RUN = {
    "id": SAMPLE_RUN_ID,
    "monitorId": SAMPLE_MONITOR_ID,
    "status": "completed",
    "output": {
        "results": [
            {
                "title": "New battery recycling facility announced",
                "url": "https://example.com/announcement",
            }
        ],
        "content": "One new facility was announced in Nevada.",
        "grounding": [
            {
                "field": "content",
                "citations": [
                    {
                        "title": "New battery recycling facility announced",
                        "url": "https://example.com/announcement",
                    }
                ],
                "confidence": "high",
            }
        ],
    },
    "failReason": None,
    "startedAt": "2026-10-28T20:00:01.000Z",
    "completedAt": "2026-10-28T20:00:09.000Z",
    "failedAt": None,
    "cancelledAt": None,
    "durationMs": 8000,
    "createdAt": "2026-10-28T20:00:00.000Z",
    "updatedAt": "2026-10-28T20:00:09.000Z",
}

SAMPLE_RUN_COMPLETED_EVENT = {
    "id": "event_01k4dbq9z2",
    "object": "event",
    "type": "monitor.run.completed",
    "data": {**SAMPLE_RUN, "metadata": {"workspace_id": "workspace_123"}},
    "createdAt": "2026-10-28T20:00:10.000Z",
}
