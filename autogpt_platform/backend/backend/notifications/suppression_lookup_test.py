"""The Postmark suppression lookup sign-up and Resend wait on.

It reads the auth stream's suppression list, and answers "not suppressed"
whenever Postmark can't answer in time, so the mailer being down never blocks a
sign-up. Postmark here is a local HTTP server, so the request postmarker builds
and the timeout it applies are the real ones.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import parse_qs, urlparse

import pytest

from backend.notifications import email as delivery
from backend.notifications.email import SUPPRESSION_LOOKUP_TIMEOUT_SECONDS, EmailSender

ADDRESS = "someone@example.com"


@pytest.mark.parametrize(
    "listed, suppressed",
    [([{"EmailAddress": "Someone@Example.com"}], True), ([], False)],
)
def test_the_lookup_reads_the_stream_suppression_list(postmark, listed, suppressed):
    postmark.answer = lambda: (200, {"Suppressions": listed})

    assert postmark.sender.is_suppressed(ADDRESS) is suppressed
    assert postmark.requests == [
        ("/message-streams/outbound/suppressions/dump", {"EmailAddress": [ADDRESS]})
    ]


def test_the_lookup_says_not_suppressed_when_postmark_errors(postmark):
    postmark.answer = lambda: (500, {"ErrorCode": 500, "Message": "Internal"})

    assert postmark.sender.is_suppressed(ADDRESS) is False


def test_the_lookup_gives_up_long_before_a_send_would(postmark):
    def late_suppressed():
        time.sleep(SUPPRESSION_LOOKUP_TIMEOUT_SECONDS + 1)
        return 200, {"Suppressions": [{"EmailAddress": ADDRESS}]}

    postmark.answer = late_suppressed

    assert postmark.sender.is_suppressed(ADDRESS) is False


class _Postmark:
    def __init__(self, sender: EmailSender, url: str):
        self.sender = sender
        self.url = url
        self.requests: list[tuple[str, dict[str, list[str]]]] = []
        self.answer: Callable[[], tuple[int, dict]] = lambda: (200, {})


@pytest.fixture
def postmark(monkeypatch):
    monkeypatch.setattr(
        delivery.settings.secrets, "postmark_server_api_token", "test-token"
    )
    monkeypatch.setattr(
        delivery.settings.config, "postmark_transactional_stream", "outbound"
    )
    stub: _Postmark

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            stub.requests.append((url.path, parse_qs(url.query)))
            status, body = stub.answer()
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    sender = EmailSender()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    assert sender.postmark_lookups
    sender.postmark_lookups.root_api_url = url
    stub = _Postmark(sender, url)
    yield stub
    server.shutdown()
    server.server_close()
