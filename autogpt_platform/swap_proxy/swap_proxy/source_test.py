import asyncio
import json

import httpx
import pytest

from swap_proxy import source as source_module
from swap_proxy.settings import Settings
from swap_proxy.source import BackendCredentialSource, SourceUnavailable

TOKEN = "ghp_a-real-looking-token"
KEY = "internal-key-0123456789abcdefghijklmnop"


def _backend(handler):
    calls: list[tuple[str, dict]] = []

    def transport(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        calls.append((request.url.path, body))
        return handler(request.url.path, body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    return BackendCredentialSource("http://backend:8012/", client), calls


def _ok(path, body):
    if path == "/get_swap_bindings":
        return httpx.Response(200, json={"github": ["github.com", "api.github.com"]})
    if body["user_id"] != "user-a":
        return httpx.Response(200, content=b"null")  # what FastAPI sends for None
    return httpx.Response(
        200,
        json={
            "name": "github",
            "values": {"access_token": TOKEN},
            "allowed_hosts": ["github.com", "api.github.com"],
        },
    )


def _authorization_seen(api_key: str = ""):
    seen: list[str | None] = []

    def transport(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return _ok(request.url.path, json.loads(request.content or b"{}"))

    client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    return BackendCredentialSource("http://backend:8012/", client, api_key), seen


async def test_every_call_carries_the_internal_key():
    source, seen = _authorization_seen(KEY)
    await source.bound_names("api.github.com")
    await source.resolve("user-a", "github", "api.github.com", "box-1")
    assert seen == [f"Bearer {KEY}", f"Bearer {KEY}"]


async def test_without_a_key_no_authorization_is_sent():
    source, seen = _authorization_seen()
    await source.bound_names("api.github.com")
    await source.resolve("user-a", "github", "api.github.com", "box-1")
    assert seen == [None, None]


async def test_the_key_is_the_backends_own_setting(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_KEY", KEY)
    seen: list[str | None] = []

    def transport(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return _ok(request.url.path, {})

    client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    source = BackendCredentialSource.from_settings(Settings(), client)
    await source.bound_names("api.github.com")
    assert seen == [f"Bearer {KEY}"]


@pytest.mark.parametrize("key", ["", "too-short", "a key with spaces " * 3, "é" * 40])
def test_a_missing_or_malformed_key_stops_the_proxy_starting(monkeypatch, key):
    monkeypatch.setenv("INTERNAL_API_KEY", key)
    with pytest.raises(ValueError, match="INTERNAL_API_KEY"):
        BackendCredentialSource.from_settings(Settings())


def test_an_unset_key_stops_the_proxy_starting(monkeypatch):
    """Every lookup would be refused, while the proxy looked healthy."""
    monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="INTERNAL_API_KEY"):
        BackendCredentialSource.from_settings(Settings())


async def test_an_environment_proxy_never_sees_a_call(monkeypatch):
    """The calls are plain HTTP and carry the key: an outbound proxy set in the
    environment must not be where they go."""
    reached: list[bytes] = []

    async def trap(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        reached.append(await reader.read(1024))
        writer.close()

    server = await asyncio.start_server(trap, "127.0.0.1", 0)
    proxy = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, proxy)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    # A host that never resolves: going direct fails fast, a proxy would be asked.
    source = BackendCredentialSource("http://swap-service.invalid:8012", api_key=KEY)
    async with server:
        with pytest.raises(SourceUnavailable):
            await source.bindings()
    await source.aclose()
    assert reached == []


async def test_bound_names_come_from_the_bindings_table_asked_once():
    source, calls = _backend(_ok)
    assert await source.bound_names("api.github.com") == {"github"}
    assert await source.bound_names("API.GitHub.com:443") == {"github"}
    assert await source.bound_names("example.com") == set()
    assert [path for path, _ in calls] == ["/get_swap_bindings"]


async def test_resolve_names_the_user_and_the_host_and_is_cached_briefly():
    source, calls = _backend(_ok)
    credential = None
    for _ in range(3):
        credential = await source.resolve("user-a", "github", "api.github.com", "box-1")
    assert credential is not None
    assert credential.values == {"access_token": TOKEN}
    assert credential.allowed_hosts == ("github.com", "api.github.com")
    assert calls == [
        (
            "/resolve_swap_credential",
            {
                "user_id": "user-a",
                "name": "github",
                "host": "api.github.com",
                "box": "box-1",
            },
        )
    ]


async def test_one_users_answer_is_never_anothers():
    source, _ = _backend(_ok)
    assert (
        await source.resolve("user-a", "github", "api.github.com", "box-1") is not None
    )
    assert await source.resolve("user-b", "github", "api.github.com", "box-1") is None


async def test_one_boxs_answer_is_never_anothers():
    """The backend answers per box (its ceiling, whether it is still live), so
    a cached answer for one box is not served to another of the same user's."""
    source, calls = _backend(_ok)
    await source.resolve("user-a", "github", "api.github.com", "box-1")
    await source.resolve("user-a", "github", "api.github.com", "box-2")
    assert [body["box"] for _, body in calls] == ["box-1", "box-2"]


async def test_the_cache_expires(monkeypatch):
    source, calls = _backend(_ok)
    now = [1000.0]
    monkeypatch.setattr(source_module.time, "monotonic", lambda: now[0])
    await source.resolve("user-a", "github", "api.github.com", "box-1")
    now[0] += source_module._CREDENTIAL_TTL + 1
    await source.resolve("user-a", "github", "api.github.com", "box-1")
    assert len(calls) == 2


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, json={"type": "RuntimeError"}),
        httpx.Response(404, json={"type": "NoLiveBox"}),  # a stale or foreign box
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json={"name": "github"}),  # no values
    ],
)
async def test_anything_but_a_clean_answer_is_unavailable_and_not_cached(response):
    source, calls = _backend(lambda path, body: response)
    for _ in range(2):
        with pytest.raises(SourceUnavailable):
            await source.resolve("user-a", "github", "api.github.com", "box-1")
    assert len(calls) == 2


async def test_a_failed_bindings_refresh_keeps_the_previous_table(monkeypatch):
    healthy = [True]

    def handler(path, body):
        return _ok(path, body) if healthy[0] else httpx.Response(503)

    source, _ = _backend(handler)
    now = [1000.0]
    monkeypatch.setattr(source_module.time, "monotonic", lambda: now[0])
    assert await source.bound_names("github.com") == {"github"}
    healthy[0] = False
    now[0] += source_module._BINDINGS_TTL + 1
    assert await source.bound_names("github.com") == {"github"}


async def test_after_a_failed_refresh_the_backend_is_not_asked_on_every_lookup(
    monkeypatch,
):
    """Each attempt can take the whole timeout; during an outage that would
    be added to every request."""
    healthy = [True]

    def handler(path, body):
        return _ok(path, body) if healthy[0] else httpx.Response(503)

    source, calls = _backend(handler)
    now = [1000.0]
    monkeypatch.setattr(source_module.time, "monotonic", lambda: now[0])
    await source.bound_names("github.com")
    healthy[0] = False
    now[0] += source_module._BINDINGS_TTL + 1
    for _ in range(5):
        assert await source.bound_names("github.com") == {"github"}
    assert len(calls) == 2  # the first table, one failed refresh
    # Asked again once the retry interval has passed, and back on a good answer.
    healthy[0] = True
    now[0] += source_module._BINDINGS_RETRY + 1
    assert await source.bound_names("github.com") == {"github"}
    assert len(calls) == 3
    await source.bound_names("github.com")
    assert len(calls) == 3


async def test_no_table_at_all_is_unavailable():
    source, _ = _backend(lambda path, body: httpx.Response(503))
    with pytest.raises(SourceUnavailable):
        await source.bound_names("github.com")
