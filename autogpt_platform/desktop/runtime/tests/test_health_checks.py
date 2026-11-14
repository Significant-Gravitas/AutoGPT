import contextlib
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from autogpt_desktop import supervisor


@contextlib.contextmanager
def http_server(status: int):
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.01)
        )
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/health", requests
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


@pytest.mark.parametrize("source", ["environment", "system"])
@pytest.mark.parametrize("status", [200, 401, 404, 499, 500, 503])
def test_readiness_uses_local_service_even_with_host_proxy(monkeypatch, source, status):
    proxy_status = 503 if status < 500 else 200
    with (
        http_server(status) as (url, service),
        http_server(proxy_status) as (
            proxy_url,
            proxy,
        ),
    ):
        monkeypatch.setenv("no_proxy", "example.invalid")
        monkeypatch.setenv("NO_PROXY", "example.invalid")
        monkeypatch.setattr(urllib.request, "_opener", None)
        if source == "environment":
            monkeypatch.setenv("http_proxy", proxy_url)
            monkeypatch.setenv("HTTP_PROXY", proxy_url)
        else:
            monkeypatch.setattr(
                urllib.request, "getproxies", lambda: {"http": proxy_url}
            )
        environment = dict(os.environ)

        assert supervisor._http_ok(url) is (status < 500)
        assert service == ["/health"]
        assert proxy == []
        assert dict(os.environ) == environment

        with (
            contextlib.suppress(urllib.error.HTTPError),
            urllib.request.urlopen(url, timeout=3) as response,
        ):
            assert response.status == proxy_status
        assert proxy == [url]
        assert service == ["/health"]


def test_a_closed_service_is_not_ready_even_if_proxy_answers(monkeypatch):
    with http_server(200) as (url, service):
        pass
    with http_server(200) as (proxy_url, proxy):
        monkeypatch.setenv("http_proxy", proxy_url)
        monkeypatch.setenv("no_proxy", "example.invalid")
        assert not supervisor._http_ok(url)
        assert service == []
        assert proxy == []
