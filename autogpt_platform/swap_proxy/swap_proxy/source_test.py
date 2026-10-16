import json

import httpx
import pytest

from swap_proxy import source as source_module
from swap_proxy.source import BackendCredentialSource, SourceUnavailable

TOKEN = "ghp_a-real-looking-token"


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
        credential = await source.resolve("user-a", "github", "api.github.com")
    assert credential is not None
    assert credential.values == {"access_token": TOKEN}
    assert credential.allowed_hosts == ("github.com", "api.github.com")
    assert calls == [
        (
            "/resolve_swap_credential",
            {"user_id": "user-a", "name": "github", "host": "api.github.com"},
        )
    ]


async def test_one_users_answer_is_never_anothers():
    source, _ = _backend(_ok)
    assert await source.resolve("user-a", "github", "api.github.com") is not None
    assert await source.resolve("user-b", "github", "api.github.com") is None


async def test_the_cache_expires(monkeypatch):
    source, calls = _backend(_ok)
    now = [1000.0]
    monkeypatch.setattr(source_module.time, "monotonic", lambda: now[0])
    await source.resolve("user-a", "github", "api.github.com")
    now[0] += source_module._CREDENTIAL_TTL + 1
    await source.resolve("user-a", "github", "api.github.com")
    assert len(calls) == 2


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, json={"type": "RuntimeError"}),
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json={"name": "github"}),  # no values
    ],
)
async def test_anything_but_a_clean_answer_is_unavailable_and_not_cached(response):
    source, calls = _backend(lambda path, body: response)
    for _ in range(2):
        with pytest.raises(SourceUnavailable):
            await source.resolve("user-a", "github", "api.github.com")
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
