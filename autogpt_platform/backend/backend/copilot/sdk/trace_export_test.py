import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from langfuse import Langfuse
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.sdk.trace import TracerProvider
from requests.exceptions import ConnectionError

from backend.copilot.sdk.service import _setup_langfuse_otel
from backend.copilot.sdk.trace_export import create_trace_export_session


class _TraceServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _TraceHandler)
        self.bodies: list[bytes] = []
        self.accepted: list[bytes] = []
        self.fail_requests = 1
        self.response_delay = 1.2
        self.response_status = 200


class _TraceHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        receiver = cast(_TraceServer, self.server)
        body = self.rfile.read(int(self.headers["Content-Length"]))
        receiver.bodies.append(body)
        status = receiver.response_status
        if len(receiver.bodies) <= receiver.fail_requests:
            time.sleep(receiver.response_delay)
            status = 504
        elif status == 200:
            receiver.accepted.append(body)
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.send_header("Retry-After", "1")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def trace_receiver() -> Iterator[_TraceServer]:
    receiver = _TraceServer()
    worker = threading.Thread(target=receiver.serve_forever, daemon=True)
    worker.start()
    try:
        yield receiver
    finally:
        receiver.shutdown()
        receiver.server_close()
        worker.join()


@pytest.fixture
def langfuse_client(
    trace_receiver: _TraceServer, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Langfuse]:
    for key in (
        "OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER",
        "OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = MagicMock()
    settings.secrets.langfuse_public_key = f"pk-test-{uuid4()}"
    settings.secrets.langfuse_secret_key = "sk-test"
    settings.secrets.langfuse_host = f"http://127.0.0.1:{trace_receiver.server_port}"
    settings.secrets.langfuse_tracing_environment = "test"
    with (
        patch("backend.copilot.sdk.service._is_langfuse_configured", return_value=True),
        patch("backend.copilot.sdk.service.Settings", return_value=settings),
        patch("backend.copilot.sdk.service.configure_claude_agent_sdk"),
    ):
        _setup_langfuse_otel()

    provider = TracerProvider()
    client = Langfuse(
        public_key=settings.secrets.langfuse_public_key,
        secret_key=settings.secrets.langfuse_secret_key,
        base_url=settings.secrets.langfuse_host,
        timeout=1,
        tracer_provider=provider,
    )
    try:
        yield client
    finally:
        client.shutdown()
        provider.shutdown()


def test_langfuse_retries_timed_out_span_batch(
    trace_receiver: _TraceServer, langfuse_client: Langfuse
) -> None:
    with langfuse_client.start_as_current_span(name="copilot-sdk-turn") as span:
        span.update(input="test", output="done")
    langfuse_client.flush()

    assert len(trace_receiver.bodies) == 2
    assert trace_receiver.bodies[0] == trace_receiver.bodies[1]
    assert len(trace_receiver.accepted) == 1
    request = ExportTraceServiceRequest.FromString(trace_receiver.accepted[0])
    exported = request.resource_spans[0].scope_spans[0].spans
    assert [item.name for item in exported] == ["copilot-sdk-turn"]


def test_trace_export_session_bounds_transport_retries(
    trace_receiver: _TraceServer,
) -> None:
    trace_receiver.fail_requests = 100
    trace_receiver.response_delay = 0.1
    with create_trace_export_session() as session:
        with pytest.raises(ConnectionError):
            session.post(
                f"http://127.0.0.1:{trace_receiver.server_port}/v1/traces",
                data=b"span batch",
                timeout=0.02,
            )

    assert trace_receiver.bodies == [b"span batch"] * 3
    assert not trace_receiver.accepted


@pytest.mark.parametrize("status", [401, 429, 503])
def test_trace_export_session_leaves_status_handling_to_exporter(
    trace_receiver: _TraceServer, status: int
) -> None:
    trace_receiver.fail_requests = 0
    trace_receiver.response_status = status
    with create_trace_export_session() as session:
        response = session.post(
            f"http://127.0.0.1:{trace_receiver.server_port}/v1/traces",
            data=b"span batch",
            timeout=1,
        )

    assert response.status_code == status
    assert trace_receiver.bodies == [b"span batch"]
