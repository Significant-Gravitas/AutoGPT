"""Shared, sanitized fixtures for the Conductor blocks' inline test_mock data.

Shapes mirror the live API: a `userMessage` row whose `content.id` is the
send receipt, and an `agent` row carrying a Claude Code `assistant` event in
`content.rawPayload` tagged with the turn it belongs to.
"""

MOCK_RECEIPT_ID = "msg_1"

MOCK_PROMPT_MESSAGE = {
    "id": "row_1",
    "sessionId": "sess_1",
    "sessionIndex": 1,
    "type": "userMessage",
    "content": {
        "type": "userMessage",
        "id": MOCK_RECEIPT_ID,
        "turnId": MOCK_RECEIPT_ID,
        "message": "Run the tests",
        "state": "sent",
    },
    "receivedAt": "2026-09-26T00:00:00Z",
}

MOCK_REPLY_MESSAGE = {
    "id": "row_2",
    "sessionId": "sess_1",
    "sessionIndex": 2,
    "type": "agent",
    "content": {
        "type": "agent",
        "rawPayload": {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "All tests pass now."}],
            },
        },
        "userMessageId": MOCK_RECEIPT_ID,
        "turnId": MOCK_RECEIPT_ID,
    },
    "receivedAt": "2026-09-26T00:00:01Z",
}

WAIT_MOCK_REPLY = {
    "session_status": "idle",
    "error_message": "",
    "messages": [MOCK_REPLY_MESSAGE],
    "reply": "All tests pass now.",
    "timed_out": False,
    "truncated": False,
}
