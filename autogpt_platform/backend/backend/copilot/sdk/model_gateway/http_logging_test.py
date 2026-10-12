import logging
from unittest.mock import Mock

import autogpt_libs.logging.config
import httpx
import pytest

from backend.copilot.sdk.model_gateway.testing.canned_upstream import (
    CannedReply,
    CannedUpstream,
)
from backend.util import logging as backend_logging


@pytest.mark.parametrize("structured", [False, True])
async def test_backend_logging_hides_http_response_headers(
    structured, monkeypatch, caplog
):
    configure = Mock()
    monkeypatch.setattr(autogpt_libs.logging.config, "configure_logging", configure)
    monkeypatch.setattr(
        backend_logging, "is_structured_logging_enabled", lambda: structured
    )
    for name in ("httpx", "httpcore", "httpcore.http11", "httpcore.connection"):
        monkeypatch.setattr(logging.getLogger(name), "level", logging.NOTSET)
    caplog.set_level(logging.DEBUG)
    backend_logging.configure_logging()
    reply = CannedReply.of_json({}, **{"x-private-header": "fake-private-marker"})
    async with CannedUpstream(reply) as upstream, httpx.AsyncClient(
        trust_env=False
    ) as client:
        assert (await client.get(upstream.url)).status_code == 200
    logging.getLogger("httpcore.http11").warning("safe transport warning")
    configure.assert_called_once_with(force_cloud_logging=structured)
    assert "fake-private-marker" not in caplog.text
    assert "receive_response_headers.complete" not in caplog.text
    assert "safe transport warning" in caplog.text
    assert logging.getLogger("httpcore").level == logging.WARNING
    assert logging.getLogger("httpx").level == logging.WARNING
