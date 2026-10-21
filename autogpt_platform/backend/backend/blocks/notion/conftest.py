from collections.abc import Callable
from typing import Any

import pytest

from backend.util.request import Requests

Sent = list[tuple[str, str, Any]]


@pytest.fixture
def notion_api(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Sent]:
    """Answer each Notion request with the next payload given; returns what was sent."""

    def respond(*payloads: dict) -> Sent:
        answers = iter(payloads)
        sent: Sent = []

        async def request(_self, method: str, url: str, **kwargs: Any) -> _Response:
            sent.append((method, url, kwargs.get("json")))
            return _Response(next(answers))

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
