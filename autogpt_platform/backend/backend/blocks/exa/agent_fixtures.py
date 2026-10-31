"""
Sample Exa Agent API payloads, shaped like the live spec.

Used by the Agent and Research blocks' built-in ``test_mock`` data and by the
colocated unit tests, so both exercise the real parsers on realistic JSON.
"""

SAMPLE_RUN_ID = "agent_run_01jszdfs0052sg4jc552sg4jc5"

SAMPLE_COMPLETED_RUN: dict = {
    "id": SAMPLE_RUN_ID,
    "object": "agent_run",
    "status": "completed",
    "stopReason": "schema_satisfied",
    "createdAt": "2026-05-07T21:21:52.051Z",
    "completedAt": "2026-05-07T21:23:10.500Z",
    "request": {
        "query": "Find the top 3 AI coding assistants and their pricing",
        "effort": "medium",
    },
    "output": {
        "text": "The top AI coding assistants are ...",
        "structured": {"assistants": [{"name": "Example", "price": "$10"}]},
        "grounding": [
            {
                "field": "assistants[0].price",
                "citations": [
                    {"url": "https://example.com/pricing", "title": "Pricing"}
                ],
                "confidence": "high",
            }
        ],
    },
    "usage": {
        "agentComputeUnits": 0.6,
        "searches": 4,
        "emails": 0,
        "phoneNumbers": 0,
    },
    "costDollars": {
        "total": 0.1,
        "agentCompute": 0.06,
        "search": 0.02,
        "emails": 0,
        "phoneNumbers": 0,
    },
}

SAMPLE_QUEUED_RUN: dict = {
    **SAMPLE_COMPLETED_RUN,
    "status": "queued",
    "stopReason": None,
    "completedAt": None,
    "output": {"text": "", "structured": None, "grounding": []},
    "usage": {"agentComputeUnits": 0, "searches": 0, "emails": 0, "phoneNumbers": 0},
    "costDollars": {
        "total": 0,
        "agentCompute": 0,
        "search": 0,
        "emails": 0,
        "phoneNumbers": 0,
    },
}

SAMPLE_CANCELLED_RUN: dict = {
    **SAMPLE_QUEUED_RUN,
    "status": "cancelled",
    "stopReason": "cancelled",
    "completedAt": "2026-05-07T21:22:00.000Z",
}

SAMPLE_STOPPED_RUN: dict = {
    **SAMPLE_COMPLETED_RUN,
    "stopReason": "stopped",
}

SAMPLE_FAILED_RUN: dict = {
    **SAMPLE_QUEUED_RUN,
    "status": "failed",
    "stopReason": "error",
    "completedAt": "2026-05-07T21:22:30.000Z",
}

SAMPLE_RUN_LIST: dict = {
    "object": "list",
    "data": [SAMPLE_COMPLETED_RUN],
    "hasMore": True,
    "nextCursor": "agent_run_01jszdfs0052sg4jc552sg4jc4",
}

SAMPLE_EVENT_LIST: dict = {
    "object": "list",
    "data": [
        {
            "id": "1",
            "event": "agent_run.created",
            "data": {"id": SAMPLE_RUN_ID, "status": "queued"},
            "createdAt": "2026-05-07T21:21:52.051Z",
        },
        {
            "id": "2",
            "event": "agent_run.started",
            "data": {"id": SAMPLE_RUN_ID, "status": "running"},
            "createdAt": "2026-05-07T21:21:53.000Z",
        },
    ],
    "hasMore": False,
    "nextCursor": None,
}

SAMPLE_FAILED_EVENT_LIST: dict = {
    "object": "list",
    "data": [
        {
            "id": "3",
            "event": "agent_run.failed",
            "data": {
                "id": SAMPLE_RUN_ID,
                "status": "failed",
                "error": {"code": "INTERNAL_ERROR", "message": "Upstream timeout"},
            },
            "createdAt": "2026-05-07T21:22:30.000Z",
        }
    ],
    "hasMore": False,
    "nextCursor": None,
}

SAMPLE_DELETED_RUN: dict = {
    "id": SAMPLE_RUN_ID,
    "object": "agent_run.deleted",
    "deleted": True,
}
