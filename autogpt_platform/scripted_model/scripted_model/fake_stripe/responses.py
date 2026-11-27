"""What the fake Stripe answers: JSON, an HTML page or a redirect."""

import html
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes = b""
    content_type: str = "application/json"
    location: str | None = None

    @classmethod
    def json(cls, status: int, payload: Any) -> "Response":
        return cls(status, json.dumps(payload).encode())

    @classmethod
    def html(cls, status: int, title: str, body: str) -> "Response":
        page = (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f"<title>{html.escape(title)} - fake Stripe</title></head>"
            f"<body><h1>{html.escape(title)}</h1>{body}</body></html>"
        )
        return cls(status, page.encode(), "text/html; charset=utf-8")

    @classmethod
    def redirect(cls, location: str) -> "Response":
        return cls(303, b"", "text/plain", location)
