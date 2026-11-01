"""Response compression, as the REST API registers it, has a bounded cost."""

import asyncio
import gzip
from collections.abc import Iterator

import pytest
import pytest_mock
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from backend.api.rest_api import app as rest_app

MIB = 1024 * 1024


def _registered_gzip_middleware() -> Middleware:
    entries = [m for m in rest_app.user_middleware if issubclass(m.cls, GZipMiddleware)]
    assert len(entries) == 1
    return entries[0]


def _sized(request: Request) -> Response:
    return Response(b"a" * int(request.path_params["size"]), media_type="text/plain")


def _streamed(request: Request) -> StreamingResponse:
    size, chunks = int(request.path_params["size"]), int(request.path_params["chunks"])

    def body() -> Iterator[bytes]:
        for _ in range(chunks):
            yield b"a" * size

    return StreamingResponse(body(), media_type="text/plain")


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = Starlette(
        routes=[
            Route("/sized/{size}", _sized),
            Route("/streamed/{size}/{chunks}", _streamed),
        ],
        middleware=[_registered_gzip_middleware()],
    )
    with TestClient(app, headers={"Accept-Encoding": "gzip"}) as test_client:
        yield test_client


@pytest.fixture
def compressed_bytes(mocker: pytest_mock.MockerFixture) -> list[int]:
    """How many bytes went through the compressor, one entry per write."""
    sizes: list[int] = []
    real_write = gzip.GzipFile.write

    def write(self: gzip.GzipFile, data: bytes) -> int:
        sizes.append(len(data))
        return real_write(self, data)

    mocker.patch.object(gzip.GzipFile, "write", write)
    return sizes


@pytest.fixture
def compress_levels(mocker: pytest_mock.MockerFixture) -> list[int]:
    levels: list[int] = []
    real_init = gzip.GzipFile.__init__

    def init(self: gzip.GzipFile, *args, compresslevel: int = 9, **kwargs) -> None:
        levels.append(compresslevel)
        real_init(self, *args, compresslevel=compresslevel, **kwargs)

    mocker.patch.object(gzip.GzipFile, "__init__", init)
    return levels


def test_a_response_over_the_cap_is_sent_uncompressed(
    client: TestClient, compressed_bytes: list[int]
) -> None:
    response = client.get(f"/sized/{6 * MIB}")

    assert response.status_code == 200
    assert "content-encoding" not in response.headers
    assert len(response.content) == 6 * MIB
    assert sum(compressed_bytes) == 0


def test_a_streamed_response_whose_first_chunk_is_over_the_cap_is_sent_uncompressed(
    client: TestClient, compressed_bytes: list[int]
) -> None:
    response = client.get(f"/streamed/{6 * MIB}/1")

    assert "content-encoding" not in response.headers
    assert len(response.content) == 6 * MIB
    assert sum(compressed_bytes) == 0


def test_a_streamed_response_without_a_length_that_passes_the_cap_is_sent_uncompressed(
    client: TestClient, compressed_bytes: list[int]
) -> None:
    response = client.get(f"/streamed/{MIB}/6")

    assert "content-encoding" not in response.headers
    assert response.content == b"a" * 6 * MIB
    assert sum(compressed_bytes) == 0


def test_a_streamed_response_without_a_length_within_the_cap_is_compressed(
    client: TestClient, compressed_bytes: list[int]
) -> None:
    response = client.get(f"/streamed/{MIB}/4")

    assert response.headers["content-encoding"] == "gzip"
    assert response.content == b"a" * 4 * MIB
    assert sum(compressed_bytes) == 4 * MIB
    assert max(compressed_bytes) == MIB


def test_a_mid_sized_response_is_compressed_at_a_cheap_level(
    client: TestClient, compressed_bytes: list[int], compress_levels: list[int]
) -> None:
    response = client.get(f"/sized/{MIB}")

    assert response.headers["content-encoding"] == "gzip"
    assert response.content == b"a" * MIB
    assert sum(compressed_bytes) == MIB
    assert compress_levels == [6]


def test_a_streamed_response_in_small_chunks_is_still_compressed(
    client: TestClient, compressed_bytes: list[int]
) -> None:
    response = client.get("/streamed/65536/4")

    assert response.headers["content-encoding"] == "gzip"
    assert response.content == b"a" * 65536 * 4
    assert max(compressed_bytes) == 65536


def test_a_small_response_is_sent_uncompressed(
    client: TestClient, compressed_bytes: list[int]
) -> None:
    response = client.get("/sized/1000")

    assert "content-encoding" not in response.headers
    assert sum(compressed_bytes) == 0


def _declared_but_streamed(request: Request) -> StreamingResponse:
    size, chunks = int(request.path_params["size"]), int(request.path_params["chunks"])

    def body() -> Iterator[bytes]:
        for _ in range(chunks):
            yield b"a" * size

    return StreamingResponse(
        body(),
        media_type="text/plain",
        headers={"Content-Length": str(size * chunks)},
    )


def test_a_response_whose_declared_length_is_over_the_cap_is_sent_uncompressed(
    compressed_bytes: list[int],
) -> None:
    app = Starlette(
        routes=[Route("/declared/{size}/{chunks}", _declared_but_streamed)],
        middleware=[_registered_gzip_middleware()],
    )
    with TestClient(app, headers={"Accept-Encoding": "gzip"}) as client:
        response = client.get(f"/declared/{MIB}/6")

    assert "content-encoding" not in response.headers
    assert len(response.content) == 6 * MIB
    assert sum(compressed_bytes) == 0


def test_a_response_of_exactly_the_cap_is_still_compressed(
    client: TestClient, compressed_bytes: list[int]
) -> None:
    response = client.get(f"/sized/{5 * MIB}")

    assert response.headers["content-encoding"] == "gzip"
    assert len(response.content) == 5 * MIB
    assert sum(compressed_bytes) == 5 * MIB


@pytest.mark.asyncio
async def test_an_event_stream_without_a_length_reaches_the_client_before_it_ends() -> (
    None
):
    first_chunk = b"data: first\n\n"
    delivered = asyncio.Event()
    sent: list[Message] = []

    async def stream(scope: Scope, receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        await send(
            {"type": "http.response.body", "body": first_chunk, "more_body": True}
        )
        await asyncio.wait_for(delivered.wait(), timeout=2)
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            delivered.set()

    scope: Scope = {
        "type": "http",
        "method": "GET",
        "path": "/events",
        "headers": [(b"accept-encoding", b"gzip")],
    }
    entry = _registered_gzip_middleware()
    middleware = entry.cls(stream, *entry.args, **entry.kwargs)
    await middleware(scope, receive, send)

    start = next(m for m in sent if m["type"] == "http.response.start")
    assert b"content-encoding" not in dict(start["headers"])
    bodies = [m["body"] for m in sent if m["type"] == "http.response.body"]
    assert bodies[0] == first_chunk
