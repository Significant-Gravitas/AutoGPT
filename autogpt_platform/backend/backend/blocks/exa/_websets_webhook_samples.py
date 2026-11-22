"""
Sample Websets webhook, attempt and event responses, shaped like the API spec.

Used as block test mocks and as fixtures for the unit tests.
"""

SAMPLE_WEBHOOK_ID = "webhook_01jz2h3k4m5n6p7q8r9s0t1u2v"
SAMPLE_EVENT_ID = "event_01jz2h3k4m5n6p7q8r9s0t1u2w"

SAMPLE_WEBHOOK = {
    "id": SAMPLE_WEBHOOK_ID,
    "object": "webhook",
    "status": "active",
    "events": ["webset.item.created", "webset.idle"],
    "url": "https://example.com/webhooks/exa",
    "secret": None,
    "metadata": {"team": "research"},
    "createdAt": "2026-10-28T12:00:00.000Z",
    "updatedAt": "2026-10-28T12:00:00.000Z",
}

SAMPLE_CREATED_WEBHOOK = {
    **SAMPLE_WEBHOOK,
    "secret": "whsec_sample_websets_secret",  # pragma: allowlist secret
}

SAMPLE_ATTEMPT = {
    "id": "attempt_01jz2h3k4m5n6p7q8r9s0t1u2x",
    "object": "webhook_attempt",
    "eventId": SAMPLE_EVENT_ID,
    "eventType": "webset.item.created",
    "webhookId": SAMPLE_WEBHOOK_ID,
    "url": "https://example.com/webhooks/exa",
    "successful": False,
    "responseHeaders": {"content-type": "text/plain"},
    "responseBody": "Internal Server Error",
    "responseStatusCode": 500,
    "attempt": 2,
    "attemptedAt": "2026-10-28T12:01:00.000Z",
}

SAMPLE_EVENT = {
    "id": SAMPLE_EVENT_ID,
    "object": "event",
    "type": "webset.item.created",
    "data": {
        "id": "witem_01jz2h3k4m5n6p7q8r9s0t1u2y",
        "object": "webset_item",
        "websetId": "webset_01jz2h3k4m5n6p7q8r9s0t1u2z",
    },
    "createdAt": "2026-10-28T12:00:30.000Z",
}
