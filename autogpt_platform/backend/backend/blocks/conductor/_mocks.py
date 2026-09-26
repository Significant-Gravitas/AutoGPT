"""Shared test fixtures for the Conductor blocks' inline test_mock data."""

WAIT_MOCK_REPLY = {
    "session_status": "idle",
    "error_message": "",
    "messages": [
        {
            "id": "msg_2",
            "sessionId": "sess_1",
            "sessionIndex": 2,
            "type": "assistant",
            "content": "All tests pass now.",
            "receivedAt": "2026-09-26T00:00:00Z",
        }
    ],
    "reply": "All tests pass now.",
    "timed_out": False,
}
