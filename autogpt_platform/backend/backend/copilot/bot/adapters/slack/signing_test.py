"""Tests for Slack request signature verification."""

import hashlib
import hmac
import time
from unittest.mock import patch

from . import signing

_SECRET = "test-signing-secret"


def _sign(body: bytes, timestamp: str) -> str:
    basestring = f"v0:{timestamp}:{body.decode()}".encode()
    digest = hmac.new(_SECRET.encode(), basestring, hashlib.sha256).hexdigest()
    return f"v0={digest}"


class TestVerify:
    def test_missing_timestamp_fails_closed(self):
        # A malformed request (no headers) must return False, not raise.
        assert signing.verify(b"{}", "", "v0=abc") is False

    def test_missing_signature_fails_closed(self):
        assert signing.verify(b"{}", str(int(time.time())), "") is False

    def test_non_numeric_timestamp_fails_closed(self):
        assert signing.verify(b"{}", "not-a-number", "v0=abc") is False

    def test_valid_signature_passes(self):
        body = b'{"type":"event_callback"}'
        ts = str(int(time.time()))
        with patch.object(signing.config, "get_signing_secret", return_value=_SECRET):
            assert signing.verify(body, ts, _sign(body, ts)) is True

    def test_wrong_signature_fails(self):
        body = b'{"type":"event_callback"}'
        ts = str(int(time.time()))
        with patch.object(signing.config, "get_signing_secret", return_value=_SECRET):
            assert signing.verify(body, ts, "v0=deadbeef") is False
