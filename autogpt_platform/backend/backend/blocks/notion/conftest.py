import json
from collections.abc import Callable, Sequence
from http import HTTPStatus
from typing import Any

import pytest

from backend.util.request import Requests, http_status_error

Sent = list[tuple[str, str, Any]]


@pytest.fixture
def notion_api(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Sent]:
    """Answer each Notion request with the next payload given; returns what was sent.

    `statuses` gives each answer's HTTP status (200 when absent); 400 or more fails
    that request the way `Requests` fails it.
    """

    def respond(*payloads: dict, statuses: Sequence[int] = ()) -> Sent:
        answers = iter(zip(payloads, [*statuses, *[200] * len(payloads)]))
        sent: Sent = []

        async def request(_self, method: str, url: str, **kwargs: Any) -> _Response:
            sent.append((method, url, kwargs.get("json")))
            payload, status = next(answers)
            if status >= 400:
                body = json.dumps(payload).encode()
                raise http_status_error(status, HTTPStatus(status).phrase, body)
            return _Response(payload)

        monkeypatch.setattr(Requests, "request", request)
        return sent

    return respond


class _Response:
    ok = True
    status = 200

    def __init__(self, payload: dict):
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def text(self) -> str:
        return ""
