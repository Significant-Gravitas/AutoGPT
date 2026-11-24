"""The catalog download rides out a GitHub blip and gives up on what won't
change.

Driven through ``resolve_catalog``, the call ``publish-skills-catalog``
makes. With a token, as a deploy downloads, against a fake GitHub API that
redirects the tarball to codeload as the real one does and serves a real
catalog tarball. Without one, as a self-hosted stack or CI's downloads,
against an HTTP server on this machine standing in for codeload. The waits
between attempts are noted instead of slept.
"""

import io
import tarfile
import threading
from collections.abc import Callable, Iterator
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from backend.api.features.store import skill_catalog_github
from backend.api.features.store.skill_catalog_fixture import expert_yaml, write_catalog
from backend.api.features.store.skill_catalog_github import ATTEMPTS, WAIT_BUDGET_S
from backend.api.features.store.skill_catalog_release import (
    load_release,
    resolve_catalog,
)
from backend.util import retry as retry_module

REPO = "Significant-Gravitas/skills-catalog"
SHA = "a5dc0724d9d3b51448d59eb89bf64568cb83178a"
COMMIT_PATH = f"/repos/{REPO}/commits/main"
TARBALL_PATH = f"/repos/{REPO}/tarball/{SHA}"
CODELOAD = f"https://codeload.github.com/{REPO}/legacy.tar.gz/{SHA}"
CODELOAD_MAIN = f"/{REPO}/tar.gz/main"
SLUGS = {"cold-email", "warm-intro"}

Answer = Callable[[httpx.Request], httpx.Response]


def status(code: int, *, retry_after: str | None = None, text: str = "") -> Answer:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return lambda request: httpx.Response(code, headers=headers, text=text)


def broken(error: type[httpx.TransportError]) -> Answer:
    def answer(request: httpx.Request) -> httpx.Response:
        raise error("injected", request=request)

    return answer


class FakeGitHub:
    """The two API endpoints the download calls, and codeload. An endpoint
    answers its scripted failures in turn, then as GitHub does: the commit
    endpoint with the SHA, the tarball endpoint with a redirect to codeload."""

    def __init__(self, tarball: bytes, failures: dict[str, list[Answer]]) -> None:
        self.tarball = tarball
        self.failures = failures
        self.requests: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.requests.append(path)
        if self.failures.get(path):
            return self.failures[path].pop(0)(request)
        if path == COMMIT_PATH:
            return httpx.Response(200, text=SHA)
        if path == TARBALL_PATH:
            return httpx.Response(302, headers={"Location": CODELOAD})
        if str(request.url) == CODELOAD:
            return httpx.Response(200, content=self.tarball)
        return httpx.Response(404)


class OnlyCodeload(httpx.HTTPTransport):
    """Sends a request for codeload.github.com to *port* on this machine, and
    fails the test on a request for any other host."""

    def __init__(self, port: int) -> None:
        super().__init__()
        self.port = port

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host != "codeload.github.com":
            pytest.fail(f"requested {request.url}")
        request.url = request.url.copy_with(
            scheme="http", host="127.0.0.1", port=self.port
        )
        return super().handle_request(request)


@pytest.fixture(autouse=True)
def _download_from_github(monkeypatch: pytest.MonkeyPatch) -> None:
    # CI points the backend's tests at a checkout; this file downloads. A
    # token picks the route, so each test sets its own rather than CI's.
    for name in (
        "SKILLS_CATALOG_PATH",
        "SKILLS_CATALOG_REPO",
        "SKILLS_CATALOG_REF",
        "GITHUB_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SKILLS_CATALOG_TOKEN", "test-token")


@pytest.fixture(autouse=True)
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The waits between attempts, noted instead of slept."""
    slept: list[float] = []
    monkeypatch.setattr(retry_module, "_interruptible_sleep", slept.append)
    return slept


def pack_catalog(directory: Path, root_name: str) -> bytes:
    """A catalog release packed as GitHub packs one, with ``git archive``:
    the tree under a single *root_name* directory, and the commit in the pax
    global header."""
    root = directory / root_name
    root.mkdir(parents=True)
    write_catalog(
        root,
        {"cold-email": {"references/a.md": "# a\n"}, "warm-intro": {}},
        experts={"max": expert_yaml("max", ["cold-email"])},
    )
    packed = io.BytesIO()
    with tarfile.open(
        fileobj=packed,
        mode="w:gz",
        format=tarfile.PAX_FORMAT,
        pax_headers={"comment": SHA},
    ) as archive:
        archive.add(root, arcname=root.name)
    return packed.getvalue()


@pytest.fixture
def tarball(tmp_path: Path) -> bytes:
    """The API's tarball, under one ``<owner>-<repo>-<sha>`` directory."""
    return pack_catalog(
        tmp_path / "packed", f"Significant-Gravitas-skills-catalog-{SHA[:7]}"
    )


@pytest.fixture
def github(
    monkeypatch: pytest.MonkeyPatch, tarball: bytes
) -> Callable[[dict[str, list[Answer]]], FakeGitHub]:
    """Answers the download's requests from a FakeGitHub with these failures."""

    def serve(failures: dict[str, list[Answer]]) -> FakeGitHub:
        fake = FakeGitHub(tarball, failures)
        transport = httpx.MockTransport(fake)
        monkeypatch.setattr(
            skill_catalog_github, "Client", partial(httpx.Client, transport=transport)
        )
        return fake

    return serve


@pytest.fixture
def into(tmp_path: Path) -> Path:
    directory = tmp_path / "download"
    directory.mkdir()
    return directory


@pytest.fixture
def codeload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """codeload, served from this machine: a 503, then the catalog packed as
    codeload packs a branch, under one ``<repo>-<ref>`` directory. Yields the
    paths asked for."""
    archive = pack_catalog(tmp_path / "codeload", "skills-catalog-main")
    answers = [503]
    asked: list[str] = []

    class Codeload(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            asked.append(self.path)
            code = answers.pop(0) if answers else 200
            body = archive if code == 200 else b""
            self.send_response(code)
            self.send_header("Content-Type", "application/x-gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Codeload)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    monkeypatch.setattr(
        skill_catalog_github,
        "Client",
        partial(httpx.Client, transport=OnlyCodeload(server.server_port)),
    )
    yield asked
    server.shutdown()
    serving.join()
    server.server_close()


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(status(504), id="gateway-timeout"),
        pytest.param(status(500), id="server-error"),
        pytest.param(status(429), id="too-many-requests"),
        pytest.param(broken(httpx.ConnectError), id="refused"),
        pytest.param(broken(httpx.ReadError), id="reset"),
        pytest.param(broken(httpx.ReadTimeout), id="timed-out"),
        pytest.param(broken(httpx.RemoteProtocolError), id="cut-short"),
    ],
)
def test_a_failure_that_may_pass_is_tried_again_and_the_catalog_loads(
    github, waits: list[float], into: Path, failure: Answer
):
    fake = github({TARBALL_PATH: [failure]})

    source = resolve_catalog(into)

    assert source.revision == SHA
    assert load_release(source.root).slugs == SLUGS
    assert fake.requests.count(TARBALL_PATH) == 2
    assert len(waits) == 1


def test_resolving_the_ref_is_tried_again_too(github, into: Path):
    fake = github({COMMIT_PATH: [status(502)]})

    source = resolve_catalog(into)

    assert source.revision == SHA
    assert load_release(source.root).slugs == SLUGS
    assert fake.requests.count(COMMIT_PATH) == 2


@pytest.mark.parametrize(
    ("path", "code"),
    [
        pytest.param(COMMIT_PATH, 422, id="unknown-ref"),
        pytest.param(TARBALL_PATH, 404, id="unknown-commit"),
        pytest.param(TARBALL_PATH, 401, id="bad-token"),
        pytest.param(TARBALL_PATH, 403, id="forbidden"),
    ],
)
def test_a_failure_that_stays_fails_at_once(
    github, waits: list[float], into: Path, path: str, code: int
):
    fake = github({path: [status(code)]})

    with pytest.raises(httpx.HTTPStatusError) as raised:
        resolve_catalog(into)

    assert raised.value.response.status_code == code
    assert fake.requests.count(path) == 1
    assert waits == []


def test_server_errors_past_the_last_attempt_raise_the_last_one(
    github, waits: list[float], into: Path
):
    fake = github(
        {
            TARBALL_PATH: [
                status(503, text=f"attempt {n}") for n in range(1, ATTEMPTS + 2)
            ]
        }
    )

    with pytest.raises(httpx.HTTPStatusError) as raised:
        resolve_catalog(into)

    assert raised.value.response.text == f"attempt {ATTEMPTS}"
    assert fake.requests.count(TARBALL_PATH) == ATTEMPTS
    assert len(waits) == ATTEMPTS - 1
    assert waits == sorted(waits)
    assert sum(waits) <= WAIT_BUDGET_S


@pytest.mark.parametrize("code", [429, 403, 503])
def test_a_retry_after_is_waited_out(github, waits: list[float], into: Path, code: int):
    github({TARBALL_PATH: [status(code, retry_after="7")]})

    source = resolve_catalog(into)

    assert load_release(source.root).slugs == SLUGS
    assert waits == [7]


_PAST_THE_BUDGET = int(WAIT_BUDGET_S) + 1
_OVER_HALF_THE_BUDGET = int(WAIT_BUDGET_S / 2) + 1


@pytest.mark.parametrize(
    ("asked", "attempts", "waited"),
    [
        pytest.param([_PAST_THE_BUDGET], 1, [], id="one-past-it"),
        pytest.param(
            [_OVER_HALF_THE_BUDGET] * 2,
            2,
            [_OVER_HALF_THE_BUDGET],
            id="two-that-add-up-past-it",
        ),
    ],
)
def test_a_retry_after_past_the_wait_budget_ends_the_retries(
    github,
    waits: list[float],
    into: Path,
    asked: list[int],
    attempts: int,
    waited: list[int],
):
    fake = github(
        {TARBALL_PATH: [status(429, retry_after=str(seconds)) for seconds in asked]}
    )

    with pytest.raises(httpx.HTTPStatusError) as raised:
        resolve_catalog(into)

    assert raised.value.response.status_code == 429
    assert fake.requests.count(TARBALL_PATH) == attempts
    assert waits == waited


def test_without_a_token_the_catalog_comes_from_codeload_with_its_commit(
    monkeypatch: pytest.MonkeyPatch, codeload: list[str], waits: list[float], into: Path
):
    """Nothing asks the REST API, whose limit without a token CI runners
    share; the commit is the one the archive names, and a blip is still
    ridden out."""
    monkeypatch.delenv("SKILLS_CATALOG_TOKEN")

    source = resolve_catalog(into)

    assert source.revision == SHA
    assert source.root == into / "skills-catalog-main"
    assert load_release(source.root).slugs == SLUGS
    assert codeload == [CODELOAD_MAIN, CODELOAD_MAIN]
    assert len(waits) == 1
