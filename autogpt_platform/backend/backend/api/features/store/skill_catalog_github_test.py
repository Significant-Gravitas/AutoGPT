"""The catalog download rides out a GitHub blip and gives up on what won't
change.

Driven through ``resolve_catalog``, the call ``publish-skills-catalog``
makes, against a fake GitHub that redirects the tarball to codeload as the
real one does and serves a real catalog tarball. The waits between attempts
are noted instead of slept.
"""

import io
import tarfile
from collections.abc import Callable
from functools import partial
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


@pytest.fixture(autouse=True)
def _download_from_github(monkeypatch: pytest.MonkeyPatch) -> None:
    # CI points the backend's tests at a checkout; this file downloads.
    for name in ("SKILLS_CATALOG_PATH", "SKILLS_CATALOG_REPO", "SKILLS_CATALOG_REF"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The waits between attempts, noted instead of slept."""
    slept: list[float] = []
    monkeypatch.setattr(retry_module, "_interruptible_sleep", slept.append)
    return slept


@pytest.fixture
def tarball(tmp_path: Path) -> bytes:
    """A catalog release packed as GitHub packs one: the tree under a single
    ``<owner>-<repo>-<sha>`` directory."""
    root = tmp_path / "packed" / f"Significant-Gravitas-skills-catalog-{SHA[:7]}"
    root.mkdir(parents=True)
    write_catalog(
        root,
        {"cold-email": {"references/a.md": "# a\n"}, "warm-intro": {}},
        experts={"max": expert_yaml("max", ["cold-email"])},
    )
    packed = io.BytesIO()
    with tarfile.open(fileobj=packed, mode="w:gz") as archive:
        archive.add(root, arcname=root.name)
    return packed.getvalue()


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
